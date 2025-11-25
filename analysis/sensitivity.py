# analysis/sensitivity.py

import os
import json
from typing import Dict, Any, List

import pandas as pd

from src.run_scenario import solve_scenario

RESULTS_ROOT = "results"
CONFIGS_ROOT = "configs"

BASE_CONFIG_PATH = os.path.join(CONFIGS_ROOT, "base_2023.json")
SENS_PREFIX = "sens"

SENSITIVITY_CASES: List[Dict[str, Any]] = [
    # CAPEX energy cost ±20%
    {"label": "CAPEX_-20%", "param": "capex_energy_eur_per_mwh", "multiplier": 0.8, "absolute": None},
    {"label": "CAPEX_+20%", "param": "capex_energy_eur_per_mwh", "multiplier": 1.2, "absolute": None},

    # Degradation cost per MWh ±50%
    {"label": "DegCost_-50%", "param": "degradation_cost_eur_per_mwh_throughput", "multiplier": 0.5, "absolute": None},
    {"label": "DegCost_+50%", "param": "degradation_cost_eur_per_mwh_throughput", "multiplier": 1.5, "absolute": None},

    # Discount rate ±2 percentage points
    {"label": "WACC_5%", "param": "discount_rate", "multiplier": None, "absolute": 0.05},
    {"label": "WACC_9%", "param": "discount_rate", "multiplier": None, "absolute": 0.09},

    # ATR availability: tighter / looser
    {"label": "ATR70", "param": "atr85_availability_factor", "multiplier": None, "absolute": 0.70},
    {"label": "ATR100", "param": "atr85_availability_factor", "multiplier": None, "absolute": 1.00},

    # Efficiency changes (charge and discharge)
    {"label": "Eff_0.94_ch", "param": "eta_ch", "multiplier": None, "absolute": 0.94},
    {"label": "Eff_0.94_dis", "param": "eta_dis", "multiplier": None, "absolute": 0.94},
    {"label": "Eff_0.98_ch", "param": "eta_ch", "multiplier": None, "absolute": 0.98},
    {"label": "Eff_0.98_dis", "param": "eta_dis", "multiplier": None, "absolute": 0.98},

    # Grid fee per MW per year ±50%
    {"label": "GridFee_-50%", "param": "grid_fee_eur_per_mw_per_year", "multiplier": 0.5, "absolute": None},
    {"label": "GridFee_+50%", "param": "grid_fee_eur_per_mw_per_year", "multiplier": 1.5, "absolute": None},
]


def load_base_config() -> Dict[str, Any]:
    if not os.path.exists(BASE_CONFIG_PATH):
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG_PATH}")
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_config(config: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def scenario_dir(scen_name: str) -> str:
    return os.path.join(RESULTS_ROOT, scen_name)


def load_kpis_project(scen_name: str) -> Dict[str, Any]:
    """
    Load project-level KPIs and a flag indicating if all years were solved optimally.
    """
    # Project-level KPIs
    path_project = os.path.join(scenario_dir(scen_name), "kpis_project.csv")
    if not os.path.exists(path_project):
        raise FileNotFoundError(f"{path_project} not found. Did the scenario run successfully?")
    df_proj = pd.read_csv(path_project)
    if df_proj.empty:
        raise ValueError(f"kpis_project.csv for {scen_name} is empty.")
    row = df_proj.iloc[0]

    # Per-year KPIs (for solver status)
    path_years = os.path.join(scenario_dir(scen_name), "kpis_per_year.csv")
    if not os.path.exists(path_years):
        # If missing, we can't say anything about optimality → mark as False
        all_optimal = False
        num_non_optimal_years = None
    else:
        df_years = pd.read_csv(path_years)

        if "solver_status" in df_years.columns:
            all_optimal = bool((df_years["solver_status"] == "Optimal").all())
            non_optimal_mask = df_years["solver_status"] != "Optimal"
            num_non_optimal_years = int(non_optimal_mask.sum())
        else:
            # Old runs without status column
            all_optimal = False
            num_non_optimal_years = None

    return {
        "npv_eur": float(row.get("npv_eur", 0.0)),
        "lcos_eur_per_mwh": float(row.get("lcos_eur_per_mwh", 0.0)),
        "capex_total_eur": float(row.get("capex_total_eur", 0.0)),
        "all_years_optimal": all_optimal,
        "num_non_optimal_years": num_non_optimal_years,
    }


def make_case_scenario_name(base_name: str, label: str) -> str:
    safe_label = label.replace("%", "pct").replace(" ", "")
    return f"{SENS_PREFIX}_{base_name}_{safe_label}"


def apply_sensitivity_to_config(base_cfg: Dict[str, Any],
                                case: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base_cfg)  # shallow copy is fine (scalars)

    param = case["param"]
    mult = case.get("multiplier")
    absolute = case.get("absolute")

    if param not in cfg:
        raise KeyError(f"Parameter '{param}' not found in base config.")

    base_val = cfg[param]

    if mult is not None and absolute is not None:
        raise ValueError(f"Sensitivity case {case['label']} has both multiplier and absolute set.")

    if mult is not None:
        new_val = base_val * mult
    elif absolute is not None:
        new_val = absolute
    else:
        raise ValueError(f"Sensitivity case {case['label']} has neither multiplier nor absolute set.")

    cfg[param] = new_val
    return cfg


def run_all_sensitivities() -> None:
    base_cfg = load_base_config()
    base_name = base_cfg.get("name", "base_2023")

    results_rows: List[Dict[str, Any]] = []

    # Base scenario KPIs
    base_kpis = load_kpis_project(base_name)
    results_rows.append(
        {
            "scenario": base_name,
            "case_label": "BASE",
            "param": "BASE",
            "base_value": None,
            "new_value": None,
            "npv_eur": base_kpis["npv_eur"],
            "lcos_eur_per_mwh": base_kpis["lcos_eur_per_mwh"],
            "capex_total_eur": base_kpis["capex_total_eur"],
            "all_years_optimal": base_kpis["all_years_optimal"],
            "num_non_optimal_years": base_kpis["num_non_optimal_years"],
        }
    )

    for case in SENSITIVITY_CASES:
        label = case["label"]
        param = case["param"]

        scen_name = make_case_scenario_name(base_name, label)
        cfg_case = apply_sensitivity_to_config(base_cfg, case)
        cfg_case["name"] = scen_name

        base_val = base_cfg[param]
        new_val = cfg_case[param]

        cfg_path = os.path.join(CONFIGS_ROOT, f"{scen_name}.json")
        write_config(cfg_case, cfg_path)

        print(f"\n=== Running sensitivity case: {label} ===")
        print(f"Parameter: {param}, base: {base_val}, new: {new_val}")
        print(f"Config file: {cfg_path}")
        print(f"Scenario name: {scen_name}")

        solve_scenario(cfg_path)

        kpis = load_kpis_project(scen_name)

        results_rows.append(
            {
                "scenario": scen_name,
                "case_label": label,
                "param": param,
                "base_value": base_val,
                "new_value": new_val,
                "npv_eur": kpis["npv_eur"],
                "lcos_eur_per_mwh": kpis["lcos_eur_per_mwh"],
                "capex_total_eur": kpis["capex_total_eur"],
                "all_years_optimal": kpis["all_years_optimal"],
                "num_non_optimal_years": kpis["num_non_optimal_years"],
            }
        )

    df_results = pd.DataFrame(results_rows)
    out_csv = os.path.join(RESULTS_ROOT, "sensitivity_summary.csv")
    df_results.to_csv(out_csv, index=False)

    out_tex = os.path.join(RESULTS_ROOT, "sensitivity_summary.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(df_results.to_latex(index=False, float_format="%.2f"))

    print(f"\nSaved sensitivity summary CSV to {out_csv}")
    print(f"Saved sensitivity summary LaTeX to {out_tex}")


def main() -> None:
    run_all_sensitivities()


if __name__ == "__main__":
    main()
