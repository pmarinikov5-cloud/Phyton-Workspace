# analysis/financial_tables.py

import os
import json
from typing import Dict, Any, List

import pandas as pd

RESULTS_ROOT = "results"

SCENARIOS: List[str] = [
    "base_2023",
    "base_2023_da_only",
]


def _scenario_dir(scen_name: str) -> str:
    return os.path.join(RESULTS_ROOT, scen_name)


def _load_config(scen_name: str) -> Dict[str, Any]:
    sdir = _scenario_dir(scen_name)
    candidates = [
        os.path.join(sdir, "config_used_base.json"),
        os.path.join(sdir, "config_used.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No config_used_*.json found in {sdir}")


def _load_kpis_per_year(scen_name: str) -> pd.DataFrame:
    path = os.path.join(_scenario_dir(scen_name), "kpis_per_year.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run src.run_scenario first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"kpis_per_year.csv for {scen_name} is empty.")
    return df.sort_values("year")


def _load_kpis_project(scen_name: str) -> pd.Series:
    path = os.path.join(_scenario_dir(scen_name), "kpis_project.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run src.run_scenario first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"kpis_project.csv for {scen_name} is empty.")
    return df.iloc[0]


def build_annual_financial_table(scen_name: str) -> pd.DataFrame:
    """
    For a given scenario, reconstruct per-year:
      - revenue components
      - fixed O&M, variable O&M, degradation cost, grid fees
      - net operating cashflow
      - discount factor, discounted CF, cum discounted CF (minus CAPEX at t=0)
    """
    sdir = _scenario_dir(scen_name)
    config = _load_config(scen_name)
    df_kpi = _load_kpis_per_year(scen_name).copy()

    E_batt = float(config["battery_energy_mwh"])
    capex_energy = float(config.get("capex_energy_eur_per_mwh", 0.0))
    capex_total = capex_energy * E_batt

    fixed_om_pct = float(config.get("fixed_om_pct_capex_per_year", 0.0))
    var_om_per_mwh = float(config.get("var_om_eur_per_mwh_throughput", 0.0))
    deg_cost_per_mwh = float(config.get("degradation_cost_eur_per_mwh_throughput", 0.0))
    grid_fee_per_mw_per_year = float(config.get("grid_fee_eur_per_mw_per_year", 0.0))
    P_batt = float(config["battery_power_mw"])
    r = float(config.get("discount_rate", 0.07))

    # Ensure revenue columns exist
    for col in [
        "rev_da_eur",
        "rev_afrr_cap_up_eur",
        "rev_afrr_cap_down_eur",
        "rev_afrr_act_up_eur",
        "rev_afrr_act_down_eur",
        "rev_total_eur",
        "throughput_mwh",
    ]:
        if col not in df_kpi.columns:
            df_kpi[col] = 0.0

    years = df_kpi["year"].astype(int).values

    fixed_om_annual = fixed_om_pct * capex_total
    grid_fee_annual = grid_fee_per_mw_per_year * P_batt

    net_cf_list: List[float] = []
    disc_cf_list: List[float] = []
    cum_disc_cf_list: List[float] = []

    cum_disc_cf = -capex_total  # CAPEX at t=0

    out_rows: List[Dict[str, Any]] = []

    for _, row in df_kpi.iterrows():
        year = int(row["year"])
        rev_da = float(row["rev_da_eur"])
        rev_cap_up = float(row["rev_afrr_cap_up_eur"])
        rev_cap_down = float(row["rev_afrr_cap_down_eur"])
        rev_act_up = float(row["rev_afrr_act_up_eur"])
        rev_act_down = float(row["rev_afrr_act_down_eur"])
        rev_total = float(row["rev_total_eur"])
        throughput = float(row["throughput_mwh"])

        var_om_annual = var_om_per_mwh * throughput
        deg_cost_annual = deg_cost_per_mwh * throughput

        total_opex = fixed_om_annual + var_om_annual + deg_cost_annual + grid_fee_annual
        net_cf = rev_total - total_opex

        disc_factor = (1.0 + r) ** year
        disc_cf = net_cf / disc_factor
        cum_disc_cf += disc_cf

        net_cf_list.append(net_cf)
        disc_cf_list.append(disc_cf)
        cum_disc_cf_list.append(cum_disc_cf)

        out_rows.append(
            {
                "scenario": scen_name,
                "year": year,
                "rev_da_eur": rev_da,
                "rev_afrr_cap_up_eur": rev_cap_up,
                "rev_afrr_cap_down_eur": rev_cap_down,
                "rev_afrr_act_up_eur": rev_act_up,
                "rev_afrr_act_down_eur": rev_act_down,
                "rev_total_eur": rev_total,
                "throughput_mwh": throughput,
                "fixed_om_eur": fixed_om_annual,
                "var_om_eur": var_om_annual,
                "degradation_cost_eur": deg_cost_annual,
                "grid_fee_eur": grid_fee_annual,
                "total_opex_eur": total_opex,
                "net_operating_cashflow_eur": net_cf,
                "discount_factor": disc_factor,
                "discounted_cashflow_eur": disc_cf,
                "cum_discounted_cf_after_capex_eur": cum_disc_cf,
            }
        )

    df_out = pd.DataFrame(out_rows)
    out_path = os.path.join(sdir, "financial_per_year.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[{scen_name}] Saved per-year financial table to {out_path}")
    return df_out


def build_lifetime_summary(scenarios: List[str]) -> pd.DataFrame:
    """
    Summarise lifetime indicators per scenario:
      - NPV, LCOS (from kpis_project.csv)
      - total throughput
      - total OPEX and deg cost
      - avg degradation cost per MWh
    """
    rows: List[Dict[str, Any]] = []

    for scen in scenarios:
        sdir = _scenario_dir(scen)
        df_fin_path = os.path.join(sdir, "financial_per_year.csv")
        if not os.path.exists(df_fin_path):
            raise FileNotFoundError(
                f"{df_fin_path} not found. Run build_annual_financial_table for {scen} first."
            )
        df_fin = pd.read_csv(df_fin_path)
        proj = _load_kpis_project(scen)

        total_throughput = float(df_fin["throughput_mwh"].sum())
        total_deg_cost = float(df_fin["degradation_cost_eur"].sum())
        total_opex = float(df_fin["total_opex_eur"].sum())

        if total_throughput > 0.0:
            avg_deg_cost_per_mwh = total_deg_cost / total_throughput
        else:
            avg_deg_cost_per_mwh = float("nan")

        rows.append(
            {
                "scenario": scen,
                "npv_eur": float(proj.get("npv_eur", 0.0)),
                "lcos_eur_per_mwh": float(proj.get("lcos_eur_per_mwh", 0.0)),
                "capex_total_eur": float(proj.get("capex_total_eur", 0.0)),
                "total_throughput_mwh": total_throughput,
                "total_degradation_cost_eur": total_deg_cost,
                "total_opex_eur": total_opex,
                "avg_degradation_cost_per_mwh_eur": avg_deg_cost_per_mwh,
            }
        )

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_ROOT, "financial_summary.csv")
    df_out.to_csv(out_csv, index=False)

    out_tex = os.path.join(RESULTS_ROOT, "financial_summary.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(df_out.to_latex(index=False, float_format="%.2f"))

    print(f"Saved financial summary CSV to {out_csv}")
    print(f"Saved financial summary LaTeX to {out_tex}")
    return df_out


def main() -> None:
    # Per-year tables
    for scen in SCENARIOS:
        print(f"\n=== Building annual financial table for {scen} ===")
        build_annual_financial_table(scen)

    # Lifetime summary across scenarios
    print("\n=== Building lifetime financial summary ===")
    build_lifetime_summary(SCENARIOS)


if __name__ == "__main__":
    main()
