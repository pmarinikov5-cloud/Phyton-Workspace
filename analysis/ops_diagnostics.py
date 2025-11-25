# analysis/ops_diagnostics.py

import os
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd

RESULTS_ROOT = "results"

# Scenarios to diagnose
SCENARIOS: List[str] = [
    "base_2023",          # DA + aFRR
    "base_2023_da_only",  # DA-only
]

# Year index for diagnostics (usually Year 1)
YEAR_DIAG: int = 1

# Small tolerance for "binding" checks
EPS = 1e-3


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


def _load_dispatch_year(scen_name: str, year: int) -> pd.DataFrame:
    path = os.path.join(_scenario_dir(scen_name), f"dispatch_timeseries_y{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Make sure you saved dispatch for year {year}."
        )
    df = pd.read_csv(path).sort_values("t")
    return df


def compute_operational_summary(scen_name: str, year: int) -> pd.DataFrame:
    """
    Operational summary for given scenario and year:
    - cycles_per_day (equiv. full cycles on nominal E0)
    - total_throughput [MWh/y]
    - SoC min/mean/max [MWh]
    - % hours at SoC_min / SoC_max
    - % idle hours (|P_net| < eps)
    """
    sdir = _scenario_dir(scen_name)
    config = _load_config(scen_name)
    df_kpi = _load_kpis_per_year(scen_name)
    df_disp = _load_dispatch_year(scen_name, year)

    dt = float(config.get("dt_hours", 1.0))
    E0 = float(config["battery_energy_mwh"])
    soc_min_frac = float(config.get("soc_min_frac", 0.0))
    soc_max_frac = float(config.get("soc_max_frac", 1.0))

    # Get SoH for this year (fraction)
    row_y = df_kpi.loc[df_kpi["year"] == year]
    if row_y.empty:
        raise ValueError(f"No KPI row found for year={year} in {scen_name}.")
    soh_frac = float(row_y.iloc[0].get("soh_frac", 1.0))
    E_y = E0 * soh_frac

    soc_min_bound = soc_min_frac * E_y
    soc_max_bound = soc_max_frac * E_y

    soc = df_disp["soc_mwh"].values
    p_net = df_disp["p_da_net_mw"].values

    # Throughput: same proxy as in postprocess (0.5*(|q_ch|+|q_dis|)*dt)
    q_ch = df_disp["q_ch_mw"].abs().values
    q_dis = df_disp["q_dis_mw"].abs().values
    throughput_mwh = float(0.5 * (q_ch + q_dis).sum() * dt)

    cycles_per_day = throughput_mwh / (2.0 * E0 * 365.0)

    soc_min = float(soc.min())
    soc_max = float(soc.max())
    soc_mean = float(soc.mean())

    n_hours = soc.shape[0]
    if n_hours == 0:
        raise ValueError(f"No rows in dispatch for {scen_name}, year={year}.")

    # Binding at SoC bounds
    bind_soc_min = (soc <= soc_min_bound + EPS)
    bind_soc_max = (soc >= soc_max_bound - EPS)

    pct_soc_min = 100.0 * bind_soc_min.sum() / n_hours
    pct_soc_max = 100.0 * bind_soc_max.sum() / n_hours

    # Idle hours: |P_net| < tiny threshold
    idle = (np.abs(p_net) < 1e-4)
    pct_idle = 100.0 * idle.sum() / n_hours

    df_out = pd.DataFrame(
        [
            {
                "scenario": scen_name,
                "year": year,
                "soh_frac": soh_frac,
                "E_y_mwh": E_y,
                "total_throughput_mwh": throughput_mwh,
                "cycles_per_day_equiv_on_E0": cycles_per_day,
                "soc_min_mwh": soc_min,
                "soc_mean_mwh": soc_mean,
                "soc_max_mwh": soc_max,
                "pct_hours_at_soc_min_bound": pct_soc_min,
                "pct_hours_at_soc_max_bound": pct_soc_max,
                "pct_idle_hours": pct_idle,
            }
        ]
    )

    out_path = os.path.join(sdir, f"table_ops_summary_y{year}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[{scen_name}] Saved operational summary to {out_path}")
    return df_out


def compute_constraint_binding_stats(scen_name: str, year: int) -> pd.DataFrame:
    """
    For given scenario+year, compute % of hours where:
      - converter limit binding
      - ATR85 grid limit binding
      - SoC_min / SoC_max binding
      - headroom / footroom binding wrt expected aFRR activation
    """
    sdir = _scenario_dir(scen_name)
    config = _load_config(scen_name)
    df_kpi = _load_kpis_per_year(scen_name)
    df_disp = _load_dispatch_year(scen_name, year)

    dt = float(config.get("dt_hours", 1.0))
    E0 = float(config["battery_energy_mwh"])
    P_batt = float(config["battery_power_mw"])
    P_grid_max = float(config.get("p_grid_max_mw", P_batt))
    alpha_atr = float(config.get("atr85_availability_factor", 1.0))

    activation_ratio_up = float(config.get("activation_ratio_up", 0.03))
    activation_ratio_down = float(config.get("activation_ratio_down", 0.07))

    soc_min_frac = float(config.get("soc_min_frac", 0.0))
    soc_max_frac = float(config.get("soc_max_frac", 1.0))

    # SoH and SoC bounds for this year
    row_y = df_kpi.loc[df_kpi["year"] == year]
    if row_y.empty:
        raise ValueError(f"No KPI row found for year={year} in {scen_name}.")
    soh_frac = float(row_y.iloc[0].get("soh_frac", 1.0))
    E_y = E0 * soh_frac
    soc_min_bound = soc_min_frac * E_y
    soc_max_bound = soc_max_frac * E_y

    soc = df_disp["soc_mwh"].values
    p_net = df_disp["p_da_net_mw"].values
    g_imp = df_disp["g_imp_mw"].values
    g_exp = df_disp["g_exp_mw"].values
    p_afrr_up = df_disp["p_afrr_up_mw"].values
    p_afrr_down = df_disp["p_afrr_down_mw"].values

    n_hours = soc.shape[0]
    if n_hours == 0:
        raise ValueError(f"No rows in dispatch for {scen_name}, year={year}.")

    # Converter binding: |P_net| + reserved aFRR close to P_batt
    # Approximate required aFRR power as sum of up+down reservations
    P_req_aFRR = np.abs(p_afrr_up) + np.abs(p_afrr_down)
    conv_limit = P_batt
    bind_conv = (np.abs(p_net) + P_req_aFRR >= conv_limit * (1.0 - EPS))

    # ATR85 binding: imports or exports near alpha_atr * P_grid_max
    atr_limit = alpha_atr * P_grid_max
    bind_ATR = (np.maximum(np.abs(g_imp), np.abs(g_exp)) >= atr_limit * (1.0 - EPS))

    # SoC min/max binding (same as before)
    bind_soc_min = (soc <= soc_min_bound + EPS)
    bind_soc_max = (soc >= soc_max_bound - EPS)

    # Headroom / footroom wrt expected activation:
    # E_up_req_t ≈ ρ_up * P_afrr_up * dt
    # E_down_req_t ≈ ρ_down * P_afrr_down * dt
    E_up_req = activation_ratio_up * np.abs(p_afrr_up) * dt
    E_down_req = activation_ratio_down * np.abs(p_afrr_down) * dt

    # Headroom binding where up capacity is offered
    headroom_bind = np.zeros_like(soc, dtype=bool)
    mask_up = (np.abs(p_afrr_up) > EPS)
    headroom_bind[mask_up] = (
        soc[mask_up] >= (soc_max_bound - E_up_req[mask_up] - EPS)
    )

    # Footroom binding where down capacity is offered
    footroom_bind = np.zeros_like(soc, dtype=bool)
    mask_down = (np.abs(p_afrr_down) > EPS)
    footroom_bind[mask_down] = (
        soc[mask_down] <= (soc_min_bound + E_down_req[mask_down] + EPS)
    )

    pct_conv = 100.0 * bind_conv.sum() / n_hours
    pct_ATR = 100.0 * bind_ATR.sum() / n_hours
    pct_soc_min = 100.0 * bind_soc_min.sum() / n_hours
    pct_soc_max = 100.0 * bind_soc_max.sum() / n_hours
    pct_headroom = 100.0 * headroom_bind.sum() / n_hours
    pct_footroom = 100.0 * footroom_bind.sum() / n_hours

    df_out = pd.DataFrame(
        [
            {
                "scenario": scen_name,
                "year": year,
                "pct_hours_conv_limit_binding": pct_conv,
                "pct_hours_ATR_limit_binding": pct_ATR,
                "pct_hours_SoC_min_binding": pct_soc_min,
                "pct_hours_SoC_max_binding": pct_soc_max,
                "pct_hours_headroom_binding": pct_headroom,
                "pct_hours_footroom_binding": pct_footroom,
            }
        ]
    )

    out_path = os.path.join(sdir, f"table_constraint_binding_y{year}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[{scen_name}] Saved constraint binding stats to {out_path}")
    return df_out


def main() -> None:
    rows_ops: List[pd.DataFrame] = []
    rows_bind: List[pd.DataFrame] = []

    for scen in SCENARIOS:
        print(f"\n=== Diagnostics for scenario: {scen}, year={YEAR_DIAG} ===")
        df_ops = compute_operational_summary(scen, YEAR_DIAG)
        df_bind = compute_constraint_binding_stats(scen, YEAR_DIAG)
        rows_ops.append(df_ops)
        rows_bind.append(df_bind)

    # Optional combined summaries
    df_ops_all = pd.concat(rows_ops, ignore_index=True)
    df_bind_all = pd.concat(rows_bind, ignore_index=True)

    ops_out = os.path.join(RESULTS_ROOT, f"table_ops_summary_all_y{YEAR_DIAG}.csv")
    bind_out = os.path.join(RESULTS_ROOT, f"table_constraint_binding_all_y{YEAR_DIAG}.csv")

    df_ops_all.to_csv(ops_out, index=False)
    df_bind_all.to_csv(bind_out, index=False)

    print(f"\nSaved combined operational summaries to {ops_out}")
    print(f"Saved combined constraint binding tables to {bind_out}")


if __name__ == "__main__":
    main()
