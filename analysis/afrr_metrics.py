# analysis/afrr_metrics.py

import os
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd

RESULTS_ROOT = "results"

# Focus on base DA+aFRR scenario
SCENARIO_BASE = "base_2023"

# Diagnostics year
YEAR_DIAG = 1


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


def compute_afrr_metrics(scen_name: str, year: int) -> pd.DataFrame:
    """
    Compute annual aFRR participation metrics for given scenario+year:
    - avg_cap_up / down [MW]
    - % blocks with commitment up / down
    - E_act_up / down [MWh/y]
    - E_ch_DA vs E_ch_from_down_activation [MWh/y]
    - revenue decomposition (if available in kpis_per_year)
    """
    sdir = _scenario_dir(scen_name)
    config = _load_config(scen_name)
    df_kpi = _load_kpis_per_year(scen_name)
    df_disp = _load_dispatch_year(scen_name, year)

    dt = float(config.get("dt_hours", 1.0))
    activation_ratio_up = float(config.get("activation_ratio_up", 0.03))
    activation_ratio_down = float(config.get("activation_ratio_down", 0.07))

    # Year KPI row
    row_y = df_kpi.loc[df_kpi["year"] == year]
    if row_y.empty:
        raise ValueError(f"No KPI row for year={year} in {scen_name}.")
    row_y = row_y.iloc[0]

    # aFRR capacities
    p_up = df_disp["p_afrr_up_mw"].values
    p_down = df_disp["p_afrr_down_mw"].values
    q_ch = df_disp["q_ch_mw"].clip(lower=0.0).values  # charging power (>=0)

    # Expected activation energies (same logic as financials)
    E_up_act = activation_ratio_up * p_up * dt
    E_down_act = activation_ratio_down * p_down * dt

    E_act_up_year = float(E_up_act.sum())
    E_act_down_year = float(E_down_act.sum())

    # Charging energy from DA vs from downward activation (approx)
    # Total charging energy:
    E_ch_total = float((q_ch * dt).sum())
    # Downward activation energy is an additional "charging" component
    E_ch_from_down_act = E_act_down_year
    # Remaining is DA-driven charging (approx):
    E_ch_from_DA = max(E_ch_total - E_ch_from_down_act, 0.0)

    # Block-level metrics (presence of commitments)
    if "block_id" in df_disp.columns:
        block_ids = df_disp["block_id"].values
    else:
        block_ids = np.zeros_like(p_up)

    blocks = np.unique(block_ids[~np.isnan(block_ids)])
    if blocks.size == 0:
        pct_blocks_up = 0.0
        pct_blocks_down = 0.0
    else:
        blocks_up = 0
        blocks_down = 0
        for b in blocks:
            mask = (block_ids == b)
            if np.any(p_up[mask] > 1e-6):
                blocks_up += 1
            if np.any(p_down[mask] > 1e-6):
                blocks_down += 1
        pct_blocks_up = 100.0 * blocks_up / len(blocks)
        pct_blocks_down = 100.0 * blocks_down / len(blocks)

    avg_cap_up = float(np.mean(p_up))
    avg_cap_down = float(np.mean(p_down))

    # Revenues from kpis_per_year (if present)
    rev_da = float(row_y.get("rev_da_eur", 0.0))
    rev_cap_up = float(row_y.get("rev_afrr_cap_up_eur", 0.0))
    rev_cap_down = float(row_y.get("rev_afrr_cap_down_eur", 0.0))
    rev_act_up = float(row_y.get("rev_afrr_act_up_eur", 0.0))
    rev_act_down = float(row_y.get("rev_afrr_act_down_eur", 0.0))
    rev_total = float(row_y.get("rev_total_eur", rev_da + rev_cap_up + rev_cap_down + rev_act_up + rev_act_down))

    df_out = pd.DataFrame(
        [
            {
                "scenario": scen_name,
                "year": year,
                "avg_cap_up_mw": avg_cap_up,
                "avg_cap_down_mw": avg_cap_down,
                "pct_blocks_with_commitment_up": pct_blocks_up,
                "pct_blocks_with_commitment_down": pct_blocks_down,
                "E_act_up_mwh": E_act_up_year,
                "E_act_down_mwh": E_act_down_year,
                "E_ch_total_mwh": E_ch_total,
                "E_ch_DA_mwh": E_ch_from_DA,
                "E_ch_down_activation_mwh": E_ch_from_down_act,
                "rev_da_eur": rev_da,
                "rev_afrr_cap_up_eur": rev_cap_up,
                "rev_afrr_cap_down_eur": rev_cap_down,
                "rev_afrr_act_up_eur": rev_act_up,
                "rev_afrr_act_down_eur": rev_act_down,
                "rev_total_eur": rev_total,
            }
        ]
    )

    out_path = os.path.join(sdir, f"table_afrr_metrics_y{year}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[{scen_name}] Saved aFRR metrics to {out_path}")
    return df_out


def main() -> None:
    scen = SCENARIO_BASE
    print(f"Computing aFRR metrics for scenario: {scen}, year={YEAR_DIAG}")
    compute_afrr_metrics(scen, YEAR_DIAG)


if __name__ == "__main__":
    main()
