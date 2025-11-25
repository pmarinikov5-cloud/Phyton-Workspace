from typing import Dict, Any, List, Tuple

import pulp as pl


def build_bess_model(config: Dict[str, Any],
                     data: Dict[str, Any]) -> Tuple[pl.LpProblem, Dict[str, Dict]]:
    """
    Build the MILP model for a single project year.

    Decision variables (per hour t, block b):
      - q_ch[t]   : charging power [MW] (AC side, >=0)
      - q_dis[t]  : discharging power [MW] (AC side, >=0)
      - soc[t]    : state of charge [MWh]
      - g_imp[t]  : grid import at POI [MW] (>=0)
      - g_exp[t]  : grid export at POI [MW] (>=0)
      - p_da_net[t]: signed DA net position at POI [MW] = g_exp - g_imp

      - y_ch[t], y_dis[t] : binary charge/discharge mode flags
      - y_imp[t], y_exp[t]: binary import/export mode flags

      - u_b_up[b]   : aFRR capacity offered UP [MW]  (interpreted as P_eff_up)
      - u_b_down[b] : aFRR capacity offered DOWN [MW] (interpreted as P_eff_down)
      - z_b_dir[b]  : binary block direction (1 = up, 0 = down)
      - p_afrr_req[t]: required converter margin in hour t due to aFRR [MW]

      - z_block_dev[b]: slack for end-of-block SoC deviation [MWh]

    Objective:
      Maximise annual gross revenue (DA + aFRR cap + aFRR act)
      minus throughput penalty
      minus block-end SoC deviation penalty.

    Notes / simplifications vs. full theoretical model:
      - Activation ratios ρ^↑, ρ^↓ are global constants (not time series ρ_t).
      - u_b_up / u_b_down are treated as effective MW (P_eff) directly.
      - ATR85 is implemented as a physical derating on grid exchange (|P_DA_net| ≤ α_ATR * P),
        not as a revenue scalar.
      - Degradation via SOH is exogenous per year: we enforce E_year = SOH_y * E0 in SoC bounds
        and headroom/footroom constraints, but do not include intra-year feedback.
    """

    # --- Sets and mappings ---
    T: List[int] = data["T"]  # list of hour indices, e.g. [0, 1, ..., 8759]
    B: List[int] = data["B"]  # list of 4h block indices
    hours_in_block: Dict[int, List[int]] = data["hours_in_block"]

    # Reverse mapping t -> b
    t_to_b: Dict[int, int] = {}
    for b in B:
        for t in hours_in_block.get(b, []):
            t_to_b[int(t)] = int(b)

    # --- Parameters from config ---
    dt = float(config.get("dt_hours", 1.0))

    # Power & energy
    p_batt = float(config["battery_power_mw"])
    # Year-specific usable energy (already adjusted by SOH in run_scenario, if present)
    e_year = float(config.get("battery_energy_mwh_year",
                              config.get("battery_energy_mwh")))

    # SoC bounds as fractions of usable energy
    soc_min_frac = float(config.get("soc_min_frac", 0.1))
    soc_max_frac = float(config.get("soc_max_frac", 0.9))
    soc_min_mwh = soc_min_frac * e_year
    soc_max_mwh = soc_max_frac * e_year

    # Initial SoC
    soc_init_frac = float(config.get("soc_init_frac", 0.5))
    soc_init_mwh = soc_init_frac * e_year

    # Efficiencies
    eta_ch = float(config.get("eta_ch", 0.95))
    eta_dis = float(config.get("eta_dis", 0.95))

    # ATR85 physical factor (grid availability)
    alpha_atr = float(config.get("alpha_atr", 1.0))
    p_grid_max = alpha_atr * p_batt

    # aFRR expected activation ratios (global simplification)
    activation_ratio_up = float(config.get("activation_ratio_up", 0.03))
    activation_ratio_down = float(config.get("activation_ratio_down", 0.07))

    # Regularisation weights
    lambda_throughput = float(config.get("lambda_throughput_eur_per_mwh", 0.0))
    lambda_soc_block = float(config.get("lambda_soc_block_eur_per_mwh", 0.0))

    # Block-end SoC reference (50% of usable energy)
    soc_block_ref_mwh = 0.5 * e_year

    # --- Price data from 'data' ---
    price_da = data["price_da"]  # dict or list indexed by t

    price_afrr_cap_up = data["price_afrr_cap_up"]      # keyed by b
    price_afrr_cap_down = data["price_afrr_cap_down"]  # keyed by b

    price_afrr_act_up = data["price_afrr_act_up"]      # keyed by t
    price_afrr_act_down = data["price_afrr_act_down"]  # keyed by t

    # --- Define model ---
    model = pl.LpProblem("BESS_DA_aFRR_single_year", pl.LpMaximize)

    # --- Decision variables ---

    # Hourly power / SoC / grid
    q_ch = {t: pl.LpVariable(f"q_ch_{t}", lowBound=0.0) for t in T}
    q_dis = {t: pl.LpVariable(f"q_dis_{t}", lowBound=0.0) for t in T}

    soc = {
        t: pl.LpVariable(
            f"soc_{t}",
            lowBound=soc_min_mwh,
            upBound=soc_max_mwh
        )
        for t in T
    }

    g_imp = {t: pl.LpVariable(f"g_imp_{t}", lowBound=0.0) for t in T}
    g_exp = {t: pl.LpVariable(f"g_exp_{t}", lowBound=0.0) for t in T}

    # Signed DA net schedule at POI (export - import), can be positive or negative
    p_da_net = {
        t: pl.LpVariable(f"p_da_net_{t}", lowBound=None, upBound=None)
        for t in T
    }

    # Binary mode flags: charge/discharge and import/export
    y_ch = {
        t: pl.LpVariable(f"y_ch_{t}", lowBound=0, upBound=1, cat=pl.LpBinary)
        for t in T
    }
    y_dis = {
        t: pl.LpVariable(f"y_dis_{t}", lowBound=0, upBound=1, cat=pl.LpBinary)
        for t in T
    }

    y_imp = {
        t: pl.LpVariable(f"y_imp_{t}", lowBound=0, upBound=1, cat=pl.LpBinary)
        for t in T
    }
    y_exp = {
        t: pl.LpVariable(f"y_exp_{t}", lowBound=0, upBound=1, cat=pl.LpBinary)
        for t in T
    }

    # aFRR block capacities and direction
    u_b_up = {b: pl.LpVariable(f"u_b_up_{b}", lowBound=0.0) for b in B}
    u_b_down = {b: pl.LpVariable(f"u_b_down_{b}", lowBound=0.0) for b in B}

    # Direction binary: 1 = upwards block, 0 = downwards block
    z_b_dir = {
        b: pl.LpVariable(f"z_b_dir_{b}", lowBound=0, upBound=1, cat=pl.LpBinary)
        for b in B
    }

    # Hourly required converter margin for aFRR (P_aFRR_req[t])
    p_afrr_req = {
        t: pl.LpVariable(f"p_afrr_req_{t}", lowBound=0.0)
        for t in T
    }

    # Block-end SoC deviation slack
    z_block_dev = {
        b: pl.LpVariable(f"z_block_dev_{b}", lowBound=0.0)
        for b in B
    }

    # --- Constraints ---

    # Initial SoC
    if len(T) == 0:
        raise ValueError("Time index T is empty.")
    t0 = T[0]
    model += soc[t0] == soc_init_mwh, "soc_initial"

    # SoC dynamics (stock-and-flow, no explicit activation energy in dynamics)
    # soc[t] = soc[t_prev] + eta_ch*q_ch[t]*dt - q_dis[t]*dt/eta_dis
    for idx, t in enumerate(T):
        if idx == 0:
            continue
        t_prev = T[idx - 1]
        model += (
            soc[t]
            == soc[t_prev]
            + eta_ch * q_ch[t] * dt
            - (q_dis[t] * dt) / eta_dis
        ), f"soc_balance_t{t}"

    # POI power balance and DA net definition
    for t in T:
        # G_imp - G_exp = P_dis - P_ch
        model += (
            g_imp[t] - g_exp[t] == q_dis[t] - q_ch[t]
        ), f"poi_balance_t{t}"

        # P_DA_net = G_exp - G_imp
        model += (
            p_da_net[t] == g_exp[t] - g_imp[t]
        ), f"p_da_net_def_t{t}"

        # ATR85 physical derating on grid connection: |P_DA_net| <= alpha_atr * P
        model += p_da_net[t] <= p_grid_max, f"atr85_upper_t{t}"
        model += -p_da_net[t] <= p_grid_max, f"atr85_lower_t{t}"

    # Converter power limits + mutual exclusivity for charge/discharge
    for t in T:
        # Link power to binaries
        model += q_ch[t] <= p_batt * y_ch[t], f"q_ch_bin_cap_t{t}"
        model += q_dis[t] <= p_batt * y_dis[t], f"q_dis_bin_cap_t{t}"

        # No simultaneous charge and discharge
        model += y_ch[t] + y_dis[t] <= 1, f"ch_dis_excl_t{t}"

        # Still keep absolute max power per direction (redundant but safe)
        model += q_ch[t] <= p_batt, f"q_ch_max_t{t}"
        model += q_dis[t] <= p_batt, f"q_dis_max_t{t}"

    # Import/export mutual exclusivity at POI
    for t in T:
        model += g_imp[t] <= p_grid_max * y_imp[t], f"g_imp_bin_cap_t{t}"
        model += g_exp[t] <= p_grid_max * y_exp[t], f"g_exp_bin_cap_t{t}"
        model += y_imp[t] + y_exp[t] <= 1, f"imp_exp_excl_t{t}"

    # aFRR direction exclusivity per block (using binary z_b_dir)
    # u_b_up <= z_b * P, u_b_down <= (1 - z_b) * P
    for b in B:
        model += u_b_up[b] <= z_b_dir[b] * p_batt, f"afrr_up_dir_b{b}"
        model += u_b_down[b] <= (1.0 - z_b_dir[b]) * p_batt, f"afrr_down_dir_b{b}"

    # Link p_afrr_req[t] to block-level aFRR capacities
    for t in T:
        b = t_to_b.get(int(t), None)
        if b is None:
            model += p_afrr_req[t] == 0.0, f"afrr_req_zero_t{t}"
            continue

        model += p_afrr_req[t] >= u_b_up[b], f"afrr_req_ge_up_t{t}_b{b}"
        model += p_afrr_req[t] >= u_b_down[b], f"afrr_req_ge_down_t{t}_b{b}"
        model += p_afrr_req[t] <= u_b_up[b] + u_b_down[b], f"afrr_req_le_sum_t{t}_b{b}"
        model += p_afrr_req[t] <= p_batt, f"afrr_req_le_p_t{t}"

    # Tight converter constraint:
    # |P_DA_net[t]| + P_aFRR_req[t] <= P
    for t in T:
        model += p_da_net[t] + p_afrr_req[t] <= p_batt, f"conv_pos_t{t}"
        model += -p_da_net[t] + p_afrr_req[t] <= p_batt, f"conv_neg_t{t}"

    # SoC feasibility for aFRR activation (headroom / footroom)
    for b in B:
        for t in hours_in_block.get(b, []):
            t_int = int(t)

            # Upward activation: require enough energy
            model += (
                soc[t_int] >= activation_ratio_up * u_b_up[b] * dt
            ), f"soc_headroom_up_t{t_int}_b{b}"

            # Downward activation: require enough empty room
            model += (
                e_year - soc[t_int] >= activation_ratio_down * u_b_down[b] * dt
            ), f"soc_footroom_down_t{t_int}_b{b}"

    # Block-end SoC anchoring (soft)
    for b in B:
        hours_b = hours_in_block.get(b, [])
        if not hours_b:
            continue
        t_end = max(hours_b)
        t_end_int = int(t_end)

        model += z_block_dev[b] >= soc[t_end_int] - soc_block_ref_mwh, f"soc_blk_dev_pos_b{b}"
        model += z_block_dev[b] >= -(soc[t_end_int] - soc_block_ref_mwh), f"soc_blk_dev_neg_b{b}"

    # --- Objective function ---

    obj = pl.lpSum([])

    # 1) Day-ahead arbitrage revenue: Σ_t price_da[t] * P_DA_net[t] * dt
    obj += pl.lpSum(
        price_da[int(t)] * p_da_net[t] * dt
        for t in T
    )

    # 2) aFRR capacity revenue: Σ_b (π_cap_up[b]*u_b_up[b] + π_cap_down[b]*u_b_down[b]) * block_duration
    for b in B:
        block_hours = hours_in_block.get(b, [])
        block_duration = len(block_hours) * dt
        if block_duration <= 0:
            continue

        price_cap_up_b = float(price_afrr_cap_up.get(int(b), 0.0))
        price_cap_down_b = float(price_afrr_cap_down.get(int(b), 0.0))

        obj += price_cap_up_b * u_b_up[b] * block_duration
        obj += price_cap_down_b * u_b_down[b] * block_duration

    # 3) aFRR activation revenue (expected): Σ_t [p_act_up[t]*ρ_up*u_b_up[b] + p_act_down[t]*ρ_down*u_b_down[b]] * dt
    for t in T:
        b = t_to_b.get(int(t), None)
        if b is None:
            continue

        price_act_up_t = float(price_afrr_act_up.get(int(t), 0.0))
        price_act_down_t = float(price_afrr_act_down.get(int(t), 0.0))

        obj += price_act_up_t * activation_ratio_up * u_b_up[b] * dt
        obj += price_act_down_t * activation_ratio_down * u_b_down[b] * dt

    # 4) Throughput penalty: λ_throughput * Σ_t 0.5*(q_ch + q_dis)*dt
    if lambda_throughput != 0.0:
        obj -= lambda_throughput * pl.lpSum(
            0.5 * (q_ch[t] + q_dis[t]) * dt
            for t in T
        )

    # 5) End-of-block SoC anchoring penalty: λ_soc_block * Σ_b z_block_dev[b]
    if lambda_soc_block != 0.0:
        obj -= lambda_soc_block * pl.lpSum(
            z_block_dev[b] for b in B
        )

    model += obj

    # --- Pack variables into a dict for downstream use ---
    var_dict: Dict[str, Dict] = {
        "q_ch": q_ch,
        "q_dis": q_dis,
        "soc": soc,
        "g_imp": g_imp,
        "g_exp": g_exp,
        "p_da_net": p_da_net,
        "u_b_up": u_b_up,
        "u_b_down": u_b_down,
        "z_b_dir": z_b_dir,
        "p_afrr_req": p_afrr_req,
        "z_block_dev": z_block_dev,
        "y_ch": y_ch,
        "y_dis": y_dis,
        "y_imp": y_imp,
        "y_exp": y_exp,
    }

    return model, var_dict
