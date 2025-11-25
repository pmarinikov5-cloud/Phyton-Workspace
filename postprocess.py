from typing import Dict, Any
import os
import json

import pandas as pd
import pulp as pl

from src.financials import compute_financials


def build_dispatch_dataframe(config: Dict[str, Any],
                             data: Dict[str, Any],
                             var_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a pandas DataFrame with hourly dispatch and prices.

    Columns:
        t, block_id,
        price_da_eur_mwh,
        q_ch_mw, q_dis_mw, soc_mwh,
        g_imp_mw, g_exp_mw, p_da_net_mw,
        p_afrr_up_mw, p_afrr_down_mw,
        price_act_up_eur_mwh, price_act_down_eur_mwh,
        price_cap_up_eur_mw_h, price_cap_down_eur_mw_h
    """

    T = data["T"]
    B = data["B"]
    hours_in_block = data["hours_in_block"]

    # Reverse mapping t -> b (assuming each t belongs to exactly one block)
    t_to_b: Dict[int, int] = {}
    for b in B:
        for t in hours_in_block.get(b, []):
            t_to_b[int(t)] = int(b)

    rows = []

    for t in T:
        t_int = int(t)
        b = t_to_b.get(t_int, None)

        # Core BESS variables
        q_ch_t = var_dict["q_ch"][t_int].value()
        q_dis_t = var_dict["q_dis"][t_int].value()
        soc_t = var_dict["soc"][t_int].value()

        # Import/export at POI (from the MILP)
        g_imp_t = var_dict["g_imp"][t_int].value()
        g_exp_t = var_dict["g_exp"][t_int].value()
        # Net DA position at POI (export positive)
        p_da_net_t = g_exp_t - g_imp_t

        # Prices
        price_da_t = data["price_da"][t_int]
        price_act_up_t = data["price_afrr_act_up"].get(t_int, 0.0)
        price_act_down_t = data["price_afrr_act_down"].get(t_int, 0.0)

        # aFRR per block (capacity)
        if b is not None:
            p_afrr_up_b = var_dict["u_b_up"][b].value()
            p_afrr_down_b = var_dict["u_b_down"][b].value()
            price_cap_up_b = data["price_afrr_cap_up"].get(b, 0.0)
            price_cap_down_b = data["price_afrr_cap_down"].get(b, 0.0)
        else:
            p_afrr_up_b = 0.0
            p_afrr_down_b = 0.0
            price_cap_up_b = 0.0
            price_cap_down_b = 0.0

        rows.append(
            {
                "t": t_int,
                "block_id": b,
                "price_da_eur_mwh": price_da_t,
                "q_ch_mw": q_ch_t,
                "q_dis_mw": q_dis_t,
                "soc_mwh": soc_t,
                "g_imp_mw": g_imp_t,
                "g_exp_mw": g_exp_t,
                "p_da_net_mw": p_da_net_t,
                "p_afrr_up_mw": p_afrr_up_b,
                "p_afrr_down_mw": p_afrr_down_b,
                "price_act_up_eur_mwh": price_act_up_t,
                "price_act_down_eur_mwh": price_act_down_t,
                "price_cap_up_eur_mw_h": price_cap_up_b,
                "price_cap_down_eur_mw_h": price_cap_down_b,
            }
        )

    df = pd.DataFrame(rows)
    return df


def compute_revenue_and_throughput(config: Dict[str, Any],
                                   df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute annual revenue components and energy throughput
    from the dispatch DataFrame.

    Uses:
      Rev_t^DA = p_da_t * (G_exp_t - G_imp_t)

      aFRR cap & act with EMS derating gamma_ems.

      Throughput_t ≈ 0.5 * (|q_ch| + |q_dis|) * dt
    """

    dt = config.get("dt_hours", 1.0)
    activation_ratio_up = config.get("activation_ratio_up", 0.03)
    activation_ratio_down = config.get("activation_ratio_down", 0.07)

    # EMS capture efficiency for aFRR
    gamma_ems = config.get("gamma_ems", 1.0)

    # --- DA revenue (no per-MWh grid tariffs; those are handled as fixed €/MW/year in financials) ---
    df["rev_da_eur"] = (
        df["price_da_eur_mwh"] * (df["g_exp_mw"] - df["g_imp_mw"])
    ) * dt

    # --- aFRR capacity revenue (per hour, using effective capacity) ---
    # P_eff = gamma_ems * P_block
    df["rev_cap_up_eur"] = (
        df["price_cap_up_eur_mw_h"] * gamma_ems * df["p_afrr_up_mw"] * dt
    )
    df["rev_cap_down_eur"] = (
        df["price_cap_down_eur_mw_h"] * gamma_ems * df["p_afrr_down_mw"] * dt
    )

    # --- aFRR activation revenue (expected energy, using P_eff) ---
    df["rev_act_up_eur"] = (
        df["price_act_up_eur_mwh"]
        * activation_ratio_up
        * gamma_ems
        * df["p_afrr_up_mw"]
        * dt
    )
    df["rev_act_down_eur"] = (
        df["price_act_down_eur_mwh"]
        * activation_ratio_down
        * gamma_ems
        * df["p_afrr_down_mw"]
        * dt
    )

    rev_da = float(df["rev_da_eur"].sum())
    rev_cap_up = float(df["rev_cap_up_eur"].sum())
    rev_cap_down = float(df["rev_cap_down_eur"].sum())
    rev_act_up = float(df["rev_act_up_eur"].sum())
    rev_act_down = float(df["rev_act_down_eur"].sum())

    # --- Approximate energy throughput for degradation ---
    # throughput_t ≈ 0.5 * (|q_ch| + |q_dis|) * dt
    df["throughput_mwh_t"] = 0.5 * (
        df["q_ch_mw"].abs() + df["q_dis_mw"].abs()
    ) * dt

    throughput_total_mwh = float(df["throughput_mwh_t"].sum())

    revenues = {
        "rev_da_eur": rev_da,
        "rev_afrr_cap_up_eur": rev_cap_up,
        "rev_afrr_cap_down_eur": rev_cap_down,
        "rev_afrr_act_up_eur": rev_act_up,
        "rev_afrr_act_down_eur": rev_act_down,
        "rev_total_eur": (
            rev_da + rev_cap_up + rev_cap_down + rev_act_up + rev_act_down
        ),
        "throughput_mwh": throughput_total_mwh,
    }

    return revenues


def save_results(config: Dict[str, Any],
                 data: Dict[str, Any],
                 var_dict: Dict[str, Any],
                 model: pl.LpProblem) -> None:
    """
    Save dispatch time series and KPI + financial metrics
    to results/<scenario_name>/.
    """

    scenario_name = config.get("name", "scenario")
    results_root = "results"
    scenario_dir = os.path.join(results_root, scenario_name)

    os.makedirs(scenario_dir, exist_ok=True)

    # 1. Save config_used.json
    config_path = os.path.join(scenario_dir, "config_used.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. Build and save dispatch dataframe
    df = build_dispatch_dataframe(config, data, var_dict)
    dispatch_path = os.path.join(scenario_dir, "dispatch_timeseries.csv")
    df.to_csv(dispatch_path, index=False)

    # 3. Compute revenue breakdown and throughput
    kpis = compute_revenue_and_throughput(config, df)
    kpis["objective_value"] = float(pl.value(model.objective))

    # 4. Compute financial metrics (NPV, LCOS, etc.)
    financials = compute_financials(config, kpis)

    # Merge dictionaries
    all_kpis = {**kpis, **financials}

    # 5. Save KPIs to CSV
    kpi_path = os.path.join(scenario_dir, "kpis.csv")
    kpi_df = pd.DataFrame([all_kpis])
    kpi_df.to_csv(kpi_path, index=False)

    print(f"Saved dispatch to {dispatch_path}")
    print(f"Saved KPIs to {kpi_path}")
