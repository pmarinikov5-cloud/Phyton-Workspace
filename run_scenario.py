from typing import Dict, Any, List
import os
import json

import pandas as pd
import pulp as pl

from src.data_loader import load_all_data
from src.model_bess import build_bess_model
from src.postprocess import build_dispatch_dataframe, compute_revenue_and_throughput
from src.financials import compute_financials_multi_year


def solve_scenario(config_path: str) -> None:
    # --------------------------------------------------------------
    # 1. Load base config
    # --------------------------------------------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    scenario_name = config.get("name", "scenario")
    results_root = "results"
    scenario_dir = os.path.join(results_root, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Save the base config for reference
    config_used_path = os.path.join(scenario_dir, "config_used_base.json")
    with open(config_used_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------------------
    # 2. Load market data (once; prices are assumed same across years)
    # --------------------------------------------------------------
    data: Dict[str, Any] = load_all_data(config)

    # Initial battery energy (E0) used to scale SoH
    E0 = float(config["battery_energy_mwh"])

    # --------------------------------------------------------------
    # 3. Load SOH path (year, soh_frac or soh_percent) from CSV or Excel
    # --------------------------------------------------------------
    soh_path = config.get("soh_path_file", "data/soh_path.xlsx")

    _, ext = os.path.splitext(soh_path)
    ext = ext.lower()

    if ext in [".csv"]:
        df_soh = pd.read_csv(soh_path)
    elif ext in [".xlsx", ".xls"]:
        # read first sheet by default
        df_soh = pd.read_excel(soh_path)
    else:
        raise ValueError(f"Unsupported SOH file extension for SOH path: {ext}")

    if "year" not in df_soh.columns:
        raise ValueError("SOH file must contain column 'year'.")

    # Accept either soh_frac (0–1) or soh_percent (0–100)
    if "soh_frac" in df_soh.columns:
        df_soh["soh_frac"] = df_soh["soh_frac"].astype(float)
    elif "soh_percent" in df_soh.columns:
        df_soh["soh_frac"] = df_soh["soh_percent"].astype(float) / 100.0
    else:
        raise ValueError("SOH file must contain 'soh_frac' or 'soh_percent' column.")

    # Sort and limit to project life, ignore year 0 if present
    df_soh = df_soh.sort_values("year")
    df_soh = df_soh[df_soh["year"] >= 1]

    life_cfg = int(config.get("project_life_years", len(df_soh)))
    df_soh = df_soh[df_soh["year"] <= life_cfg]

    if df_soh.empty:
        raise ValueError("No SOH entries found for the configured project_life_years.")

    # --------------------------------------------------------------
    # 4. Loop over years, solve MILP for each year with its SoH
    # --------------------------------------------------------------
    annual_kpis_list: List[Dict[str, Any]] = []

    time_limit = config.get("time_limit_sec", 1800)
    mip_gap = config.get("mip_gap", 0.001)

    for _, row in df_soh.iterrows():
        year_idx = int(row["year"])
        soh_frac = float(row["soh_frac"])
        E_year = E0 * soh_frac

        print(f"\n=== Solving year {year_idx} with SoH={soh_frac:.3f}, E_batt={E_year:.2f} MWh ===")

        # Year-specific config
        config_y = dict(config)
        config_y["battery_energy_mwh"] = E_year
        config_y["name"] = f"{scenario_name}_y{year_idx}"
        config_y["year_index"] = year_idx
        config_y["soh_frac"] = soh_frac

        # Build model
        model, var_dict = build_bess_model(config_y, data)

        # Solve with CBC
        solver = pl.PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=mip_gap,
            msg=True
        )
        model.solve(solver)

        status = pl.LpStatus[model.status]
        obj_val = float(pl.value(model.objective)) if model.status in (
            pl.LpStatusOptimal,
            pl.LpStatusNotSolved,
            pl.LpStatusInfeasible,
            pl.LpStatusUndefined,
            pl.LpStatusUnbounded,
        ) else float("nan")

        print(f"Solver status (year {year_idx}): {status}")
        print(f"Objective value (year {year_idx}): {obj_val}")

        # Build dispatch DataFrame
        df_dispatch = build_dispatch_dataframe(config_y, data, var_dict)

        # Save dispatch per year
        dispatch_path = os.path.join(
            scenario_dir, f"dispatch_timeseries_y{year_idx}.csv"
        )
        df_dispatch.to_csv(dispatch_path, index=False)
        print(f"Saved dispatch for year {year_idx} to {dispatch_path}")

        # KPI per year
        kpis_y = compute_revenue_and_throughput(config_y, df_dispatch)
        kpis_y["objective_value"] = obj_val
        kpis_y["year"] = year_idx
        kpis_y["soh_frac"] = soh_frac
        kpis_y["solver_status"] = status  # <-- NEW: store solver status per year

        annual_kpis_list.append(kpis_y)

    # --------------------------------------------------------------
    # 5. Save per-year KPIs and compute project-level financials
    # --------------------------------------------------------------
    # Save annual KPIs table
    annual_kpis_df = pd.DataFrame(annual_kpis_list).sort_values("year")
    annual_kpis_path = os.path.join(scenario_dir, "kpis_per_year.csv")
    annual_kpis_df.to_csv(annual_kpis_path, index=False)
    print(f"Saved per-year KPIs to {annual_kpis_path}")

    # Project-level financials (multi-year NPV, LCOS)
    fin = compute_financials_multi_year(config, annual_kpis_list)
    project_kpis_path = os.path.join(scenario_dir, "kpis_project.csv")
    pd.DataFrame([fin]).to_csv(project_kpis_path, index=False)
    print(f"Saved project-level KPIs to {project_kpis_path}")

    print("\n=== Project summary ===")
    print(f"NPV [EUR]: {fin['npv_eur']:.2f}")
    print(f"LCOS [EUR/MWh]: {fin['lcos_eur_per_mwh']:.2f}")
    print(f"CAPEX [EUR]: {fin['capex_total_eur']:.2f}")
    print(f"Years used: {fin.get('project_life_years_used', len(annual_kpis_list))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BESS DA + aFRR multi-year scenario.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/base_2023.json",
        help="Path to scenario config JSON",
    )
    args = parser.parse_args()

    solve_scenario(args.config)
