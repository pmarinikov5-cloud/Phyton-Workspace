from typing import Dict, Any, List


def npv(cashflows: List[float], discount_rate: float) -> float:
    """
    Net present value of a cashflow stream.
    cashflows[0] = CF at t=0 (usually negative CAPEX)
    cashflows[t] = CF at year t (t=1..N)
    """
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1.0 + discount_rate) ** t)
    return total


def discounted_annuity(value_per_year: float,
                       discount_rate: float,
                       years: int) -> float:
    """
    Present value of a constant annual payment or quantity.
    """
    if years <= 0:
        return 0.0
    total = 0.0
    for t in range(1, years + 1):
        total += value_per_year / ((1.0 + discount_rate) ** t)
    return total


def _get_project_life_years(config: Dict[str, Any], default: int = 15) -> int:
    """
    Helper: project life in years.

    Priority:
      1) config["project_life_years"] if present
      2) config["years"] if present
      3) default (15)
    """
    if "project_life_years" in config:
        return int(config["project_life_years"])
    if "years" in config:
        return int(config["years"])
    return int(default)


def compute_capex(config: Dict[str, Any]) -> float:
    """
    Compute total CAPEX from config.

    Thesis assumption:
    CAPEX = capex_energy_eur_per_mwh * E_batt
    (power-specific CAPEX ignored or embedded in energy-specific cost).
    """
    E_batt = config["battery_energy_mwh"]
    capex_e = config.get("capex_energy_eur_per_mwh", 0.0) * E_batt
    return capex_e


def _compute_annual_cost_components(config: Dict[str, Any],
                                    capex_total: float,
                                    throughput_mwh: float) -> Dict[str, float]:
    """
    Helper: compute all annual cost components for a given year,
    given total CAPEX and that year's throughput.
    Includes:
      - fixed O&M (as % of CAPEX)
      - variable O&M (€/MWh * throughput)
      - degradation cost (€/MWh * throughput)
      - grid fee (€/MW/year * p_grid_max_mw)
    """

    fixed_om_pct = config.get("fixed_om_pct_capex_per_year", 0.0)
    var_om_per_mwh = config.get("var_om_eur_per_mwh_throughput", 0.0)
    deg_cost_per_mwh = config.get("degradation_cost_eur_per_mwh_throughput", 0.0)
    grid_fee_per_mw_year = config.get("grid_fee_eur_per_mw_per_year", 0.0)

    # Power basis for grid fee: use p_grid_max_mw if present, otherwise battery_power_mw
    p_grid = config.get("p_grid_max_mw", config.get("battery_power_mw", 0.0))

    fixed_om_annual = fixed_om_pct * capex_total
    var_om_annual = var_om_per_mwh * throughput_mwh
    deg_cost_annual = deg_cost_per_mwh * throughput_mwh
    grid_fee_annual = grid_fee_per_mw_year * p_grid

    total_annual_costs = fixed_om_annual + var_om_annual + deg_cost_annual + grid_fee_annual

    return {
        "fixed_om_annual": fixed_om_annual,
        "var_om_annual": var_om_annual,
        "deg_cost_annual": deg_cost_annual,
        "grid_fee_annual": grid_fee_annual,
        "total_annual_costs": total_annual_costs,
    }


def compute_financials(config: Dict[str, Any],
                       annual_kpis: Dict[str, float]) -> Dict[str, float]:
    """
    SINGLE-YEAR version, kept for backward compatibility.

    Uses one representative operational year repeated over project life.
    """

    life = _get_project_life_years(config, default=15)
    r = float(config.get("discount_rate", 0.08))

    capex_total = compute_capex(config)

    rev_total = float(annual_kpis["rev_total_eur"])
    throughput_mwh = float(annual_kpis["throughput_mwh"])

    # All cost components incl. grid fee
    cost_components = _compute_annual_cost_components(
        config, capex_total, throughput_mwh
    )
    fixed_om_annual = cost_components["fixed_om_annual"]
    var_om_annual = cost_components["var_om_annual"]
    deg_cost_annual = cost_components["deg_cost_annual"]
    grid_fee_annual = cost_components["grid_fee_annual"]
    total_annual_costs = cost_components["total_annual_costs"]

    annual_cashflow = rev_total - total_annual_costs

    # NPV with repeated identical year
    cashflows = [-capex_total] + [annual_cashflow] * life
    project_npv = npv(cashflows, r)

    # LCOS: discounted costs / discounted energy
    discounted_costs = capex_total + discounted_annuity(
        total_annual_costs, r, life
    )
    discounted_energy = discounted_annuity(throughput_mwh, r, life)

    if discounted_energy > 0:
        lcos = discounted_costs / discounted_energy
    else:
        lcos = float("nan")

    return {
        "capex_total_eur": capex_total,
        "annual_cashflow_eur": annual_cashflow,
        "npv_eur": project_npv,
        "lcos_eur_per_mwh": lcos,
        "throughput_mwh_per_year": throughput_mwh,
        "annual_grid_fee_eur": grid_fee_annual,
        "annual_fixed_om_eur": fixed_om_annual,
        "annual_var_om_eur": var_om_annual,
        "annual_deg_cost_eur": deg_cost_annual,
        "project_life_years_used": life,
    }


def compute_financials_multi_year(config: Dict[str, Any],
                                  annual_kpis_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    MULTI-YEAR version with exogenous SOH path.

    annual_kpis_list: list of dicts, each (at minimum) containing:
        {
          "year": int,
          "rev_total_eur": float,
          "throughput_mwh": float,
          ...
        }

    Uses:
    - CAPEX at t=0 from initial energy (E0).
    - Year-specific revenues and throughput (SoH affects operations).
    - Fixed O&M, variable O&M, degradation cost, and grid fee per year.
    """

    if not annual_kpis_list:
        return {
            "capex_total_eur": 0.0,
            "npv_eur": 0.0,
            "lcos_eur_per_mwh": float("nan"),
            "project_life_years_used": 0,
        }

    # Discount rate and project life
    life_cfg = _get_project_life_years(config, default=len(annual_kpis_list))
    r = float(config.get("discount_rate", 0.08))

    # Initial CAPEX from initial battery size
    capex_total = compute_capex(config)

    # Sort KPI entries by year and truncate to project life
    kpis_sorted = sorted(annual_kpis_list, key=lambda d: d.get("year", 0))
    kpis_used = kpis_sorted[:life_cfg]

    discounted_costs = capex_total
    discounted_energy = 0.0
    cashflows: List[float] = [-capex_total]

    # For reporting (take last-year values at the end)
    last_fixed_om = 0.0
    last_var_om = 0.0
    last_deg_cost = 0.0
    last_grid_fee = 0.0

    for i, k in enumerate(kpis_used, start=1):
        rev_total = float(k["rev_total_eur"])
        throughput_mwh = float(k["throughput_mwh"])

        # Annual cost components for this year's throughput
        cost_components = _compute_annual_cost_components(
            config, capex_total, throughput_mwh
        )
        fixed_om_annual = cost_components["fixed_om_annual"]
        var_om_annual = cost_components["var_om_annual"]
        deg_cost_annual = cost_components["deg_cost_annual"]
        grid_fee_annual = cost_components["grid_fee_annual"]
        total_annual_costs = cost_components["total_annual_costs"]

        annual_cashflow = rev_total - total_annual_costs
        cashflows.append(annual_cashflow)

        disc_factor = (1.0 + r) ** i
        discounted_costs += total_annual_costs / disc_factor
        discounted_energy += throughput_mwh / disc_factor

        # track last year’s components
        last_fixed_om = fixed_om_annual
        last_var_om = var_om_annual
        last_deg_cost = deg_cost_annual
        last_grid_fee = grid_fee_annual

    project_npv = npv(cashflows, r)
    lcos = discounted_costs / discounted_energy if discounted_energy > 0 else float("nan")

    return {
        "capex_total_eur": capex_total,
        "npv_eur": project_npv,
        "lcos_eur_per_mwh": lcos,
        "project_life_years_used": len(kpis_used),
        "annual_fixed_om_eur_last": last_fixed_om,
        "annual_var_om_eur_last": last_var_om,
        "annual_deg_cost_eur_last": last_deg_cost,
        "annual_grid_fee_eur_last": last_grid_fee,
    }
