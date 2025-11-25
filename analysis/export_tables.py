import os
import json
from typing import Dict, Any, List

import pandas as pd

# ===================== USER CONFIG =====================

RESULTS_ROOT = "results"

# Scenarios for detailed yearly tables (per-year + Year1 vs last)
SCENARIOS_DETAILED: List[str] = [
    "base_2023",
    "base_2023_da_only",
]

# Scenarios for comparison table (NPV, LCOS, Year1 split)
SCENARIOS_COMPARE: List[str] = [
    "base_2023",
    "base_2023_da_only",
]

# =======================================================


def _scenario_dir(scen_name: str) -> str:
    return os.path.join(RESULTS_ROOT, scen_name)


def _load_config(scen_name: str) -> Dict[str, Any]:
    """
    Load config_used_base.json or config_used.json from a scenario folder.
    """
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


# -------------------------------------------------------
# 1) Detailed per-year table
# -------------------------------------------------------


def export_per_year_table(scen_name: str) -> None:
    """
    Export a clean per-year table with:
    year, soh_frac, revenue components, total revenue, throughput.
    """
    sdir = _scenario_dir(scen_name)
    df = _load_kpis_per_year(scen_name).copy()

    # Ensure missing columns exist (DA-only scenario will not have aFRR cols)
    needed_cols = [
        "rev_da_eur",
        "rev_afrr_cap_up_eur",
        "rev_afrr_cap_down_eur",
        "rev_afrr_act_up_eur",
        "rev_afrr_act_down_eur",
        "rev_total_eur",
        "throughput_mwh",
        "soh_frac",
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0.0

    cols_order = [
        "year",
        "soh_frac",
        "rev_da_eur",
        "rev_afrr_cap_up_eur",
        "rev_afrr_cap_down_eur",
        "rev_afrr_act_up_eur",
        "rev_afrr_act_down_eur",
        "rev_total_eur",
        "throughput_mwh",
    ]
    df_out = df[cols_order].copy()

    out_path = os.path.join(sdir, "table_per_year_revenues_throughput.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[{scen_name}] Saved per-year table to {out_path}")


# -------------------------------------------------------
# 2) Year 1 vs last year revenue split table
# -------------------------------------------------------


def export_revenue_split_y1_last(scen_name: str) -> None:
    """
    Export a compact table comparing revenue split in Year 1 and last year.
    Useful for 'revenue stacking evolution' in the thesis.
    """
    sdir = _scenario_dir(scen_name)
    df = _load_kpis_per_year(scen_name).copy()

    # Ensure columns exist
    for col in [
        "rev_da_eur",
        "rev_afrr_cap_up_eur",
        "rev_afrr_cap_down_eur",
        "rev_afrr_act_up_eur",
        "rev_afrr_act_down_eur",
        "rev_total_eur",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    row_y1 = df.iloc[0]
    row_y_last = df.iloc[-1]

    data = {
        "component": [
            "DA",
            "aFRR capacity up",
            "aFRR capacity down",
            "aFRR activation up",
            "aFRR activation down",
            "TOTAL",
        ],
        f"Year {int(row_y1['year'])} [EUR]": [
            float(row_y1["rev_da_eur"]),
            float(row_y1["rev_afrr_cap_up_eur"]),
            float(row_y1["rev_afrr_cap_down_eur"]),
            float(row_y1["rev_afrr_act_up_eur"]),
            float(row_y1["rev_afrr_act_down_eur"]),
            float(row_y1["rev_total_eur"]),
        ],
        f"Year {int(row_y_last['year'])} [EUR]": [
            float(row_y_last["rev_da_eur"]),
            float(row_y_last["rev_afrr_cap_up_eur"]),
            float(row_y_last["rev_afrr_cap_down_eur"]),
            float(row_y_last["rev_afrr_act_up_eur"]),
            float(row_y_last["rev_afrr_act_down_eur"]),
            float(row_y_last["rev_total_eur"]),
        ],
    }

    df_out = pd.DataFrame(data)

    out_path = os.path.join(sdir, "table_revenue_split_y1_vs_last.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[{scen_name}] Saved Year1 vs last-year revenue split to {out_path}")


# -------------------------------------------------------
# 3) Scenario comparison table (NPV + Year 1 split)
# -------------------------------------------------------


def export_scenario_comparison_table(scenarios: List[str]) -> None:
    """
    Export a table comparing scenarios on:
      - NPV
      - LCOS
      - CAPEX
      - Year-1 revenue total and split
    """
    rows: List[Dict[str, float]] = []

    for scen in scenarios:
        proj = _load_kpis_project(scen)
        df_year = _load_kpis_per_year(scen)
        df_year = df_year.sort_values("year")
        row_y1 = df_year.iloc[0]

        # Ensure columns exist
        for col in [
            "rev_da_eur",
            "rev_afrr_cap_up_eur",
            "rev_afrr_cap_down_eur",
            "rev_afrr_act_up_eur",
            "rev_afrr_act_down_eur",
            "rev_total_eur",
        ]:
            if col not in df_year.columns:
                row_y1[col] = 0.0

        rows.append(
            {
                "scenario": scen,
                "npv_eur": float(proj.get("npv_eur", 0.0)),
                "lcos_eur_per_mwh": float(proj.get("lcos_eur_per_mwh", 0.0)),
                "capex_total_eur": float(proj.get("capex_total_eur", 0.0)),
                "year1": int(row_y1["year"]),
                "year1_rev_total_eur": float(row_y1["rev_total_eur"]),
                "year1_rev_da_eur": float(row_y1["rev_da_eur"]),
                "year1_rev_afrr_cap_up_eur": float(row_y1.get("rev_afrr_cap_up_eur", 0.0)),
                "year1_rev_afrr_cap_down_eur": float(row_y1.get("rev_afrr_cap_down_eur", 0.0)),
                "year1_rev_afrr_act_up_eur": float(row_y1.get("rev_afrr_act_up_eur", 0.0)),
                "year1_rev_afrr_act_down_eur": float(row_y1.get("rev_afrr_act_down_eur", 0.0)),
            }
        )

    df_out = pd.DataFrame(rows)

    out_dir = RESULTS_ROOT
    os.makedirs(out_dir, exist_ok=True)

    out_path_csv = os.path.join(out_dir, "table_scenario_comparison.csv")
    df_out.to_csv(out_path_csv, index=False)
    print(f"Saved scenario comparison CSV to {out_path_csv}")

    # Optional: LaTeX-style table, but skip silently if jinja2/pandas styler
    # is not available in the environment.
    out_path_tex = os.path.join(out_dir, "table_scenario_comparison.tex")
    try:
        latex_str = df_out.to_latex(index=False, float_format="%.2f")
        with open(out_path_tex, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"Saved scenario comparison LaTeX to {out_path_tex}")
    except Exception as e:
        # Just inform and move on – CSV is the main deliverable.
        print(f"Skipping LaTeX export (missing optional dependency): {e}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------


def main() -> None:
    # Detailed tables for each scenario in SCENARIOS_DETAILED
    for scen in SCENARIOS_DETAILED:
        print(f"Exporting detailed tables for scenario: {scen}")
        export_per_year_table(scen)
        export_revenue_split_y1_last(scen)

    # Scenario comparison (NPV, LCOS, Year1 split)
    if SCENARIOS_COMPARE:
        print(f"Exporting comparison table for: {SCENARIOS_COMPARE}")
        export_scenario_comparison_table(SCENARIOS_COMPARE)
    else:
        print("SCENARIOS_COMPARE is empty; skipping comparison export.")


if __name__ == "__main__":
    main()
