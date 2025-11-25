import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: slightly nicer default style
plt.style.use("ggplot")


# ===================== USER CONFIG =====================
# Main scenario for A–D:
MAIN_SCENARIO = "base_2023"  # folder name under results/

# Scenarios for comparison plot (E):
SCENARIOS_COMPARE = [
    "base_2023",          # e.g. DA + aFRR
    "base_2023_da_only",  # e.g. DA only
]

RESULTS_ROOT = "results"
# =======================================================


def _scenario_dir(scen_name: str) -> str:
    return os.path.join(RESULTS_ROOT, scen_name)


def load_config_for_scenario(scen_name: str) -> dict:
    """
    Load config_used_*.json from scenario folder.
    Needed for cashflow reconstruction (O&M, discount rate, etc.).
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


def load_kpis_per_year(scen_name: str) -> pd.DataFrame:
    path = os.path.join(_scenario_dir(scen_name), "kpis_per_year.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run src.run_scenario first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"kpis_per_year.csv for {scen_name} is empty.")
    return df.sort_values("year")


def load_kpis_project(scen_name: str) -> pd.Series:
    path = os.path.join(_scenario_dir(scen_name), "kpis_project.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run src.run_scenario first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"kpis_project.csv for {scen_name} is empty.")
    return df.iloc[0]


# Small helper to print values on top of bars
def _add_bar_labels(ax, rects, fmt="{:,.0f}", offset=0.01):
    """
    Attach a text label above each bar displaying its height.
    offset is relative to y-range.
    """
    if not rects:
        return
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0

    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, offset * y_span),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


# -------------------------------------------------------
# A. Revenue split by product (Year 1 & last year)
# -------------------------------------------------------
def plot_revenue_split_years(scen_name: str):
    df = load_kpis_per_year(scen_name)

    # Year 1 and last year
    row_y1 = df.iloc[0]
    row_y_last = df.iloc[-1]

    def get_components(row):
        return [
            row["rev_da_eur"],
            row.get("rev_afrr_cap_up_eur", 0.0),
            row.get("rev_afrr_cap_down_eur", 0.0),
            row.get("rev_afrr_act_up_eur", 0.0),
            row.get("rev_afrr_act_down_eur", 0.0),
        ]

    labels = ["DA", "aFRR cap up", "aFRR cap down", "aFRR act up", "aFRR act down"]

    vals_y1 = get_components(row_y1)
    vals_y_last = get_components(row_y_last)

    scenarios = [f"Year {int(row_y1['year'])}", f"Year {int(row_y_last['year'])}"]
    values = pd.DataFrame(
        [vals_y1, vals_y_last],
        index=scenarios,
        columns=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(values.index))
    rects_all = []

    for col in values.columns:
        rects = ax.bar(values.index, values[col], bottom=bottom, label=col)
        rects_all.extend(rects)
        bottom = bottom + values[col].values

    ax.set_ylabel("Annual revenue [EUR]")
    ax.set_title(f"Revenue split by product – {scen_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=0)

    # Add total labels on top of bars
    totals = values.sum(axis=1).values
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    for i, total in enumerate(totals):
        ax.annotate(
            f"{total:,.0f} €",
            xy=(i, total),
            xytext=(0, 0.01 * y_span),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------
# B. SoH vs annual revenues and throughput
# -------------------------------------------------------
def plot_soh_revenue_throughput(scen_name: str):
    df = load_kpis_per_year(scen_name)

    years = df["year"]
    soh_pct = df["soh_frac"] * 100.0
    rev_total = df["rev_total_eur"]
    throughput = df["throughput_mwh"]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # ---- Top panel: SoH and throughput ----
    color_soh = "tab:blue"
    color_th = "tab:orange"

    ax_top.set_title(f"SoH, throughput, and revenue per year – {scen_name}")
    ax_top.set_ylabel("SoH [%]")
    l1 = ax_top.plot(years, soh_pct, marker="o", color=color_soh, label="SoH [%]")

    ax_top_2 = ax_top.twinx()
    ax_top_2.set_ylabel("Throughput [MWh]")
    l2 = ax_top_2.plot(
        years, throughput, marker="s", linestyle="--", color=color_th, label="Throughput [MWh]"
    )

    lines_top = l1 + l2
    labels_top = [l.get_label() for l in lines_top]
    ax_top.legend(lines_top, labels_top, loc="upper right", fontsize=8)

    # ---- Bottom panel: total revenue ----
    ax_bottom.bar(years, rev_total, label="Total revenue", alpha=0.7)
    ax_bottom.set_xlabel("Year")
    ax_bottom.set_ylabel("Revenue [EUR]")
    ax_bottom.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------
# C. Lifetime cashflow and NPV components
# -------------------------------------------------------
def compute_annual_cashflows(scen_name: str) -> tuple[pd.DataFrame, float]:
    """
    Reconstruct annual cashflows using per-year KPIs and config:
    CF_y = Rev_total_y - O&M_y - degradation_cost_y
    """
    config = load_config_for_scenario(scen_name)
    df = load_kpis_per_year(scen_name)
    df = df.copy()

    capex = float(config.get("capex_energy_eur_per_mwh", 0.0) * config["battery_energy_mwh"])

    fixed_om_pct = float(config.get("fixed_om_pct_capex_per_year", 0.0))
    var_om_per_mwh = float(config.get("var_om_eur_per_mwh_throughput", 0.0))
    deg_cost_per_mwh = float(config.get("degradation_cost_eur_per_mwh_throughput", 0.0))
    r = float(config.get("discount_rate", 0.07))

    cashflows = []
    om_costs = []
    deg_costs = []

    for _, row in df.iterrows():
        rev_total = float(row["rev_total_eur"])
        throughput = float(row["throughput_mwh"])

        fixed_om = fixed_om_pct * capex
        var_om = var_om_per_mwh * throughput
        deg_cost = deg_cost_per_mwh * throughput

        om_cost = fixed_om + var_om
        cf = rev_total - om_cost - deg_cost

        cashflows.append(cf)
        om_costs.append(om_cost)
        deg_costs.append(deg_cost)

    df["cashflow_eur"] = cashflows
    df["om_cost_eur"] = om_costs
    df["deg_cost_eur"] = deg_costs

    # Discounting and cumulative discounted CF (after CAPEX at t=0)
    disc_factors = [(1.0 + r) ** int(y) for y in df["year"]]
    df["disc_factor"] = disc_factors
    df["disc_cashflow_eur"] = df["cashflow_eur"] / df["disc_factor"]
    df["cum_disc_cashflow_eur"] = df["disc_cashflow_eur"].cumsum() - capex

    return df, capex


def plot_cashflows_and_cumulative_npv(scen_name: str):
    df, capex = compute_annual_cashflows(scen_name)
    proj_kpis = load_kpis_project(scen_name)
    npv_project = float(proj_kpis.get("npv_eur", np.nan))

    years = df["year"]
    om_cost = df["om_cost_eur"]
    deg_cost = df["deg_cost_eur"]
    rev_total = df["rev_total_eur"]
    cum_disc_cf = df["cum_disc_cashflow_eur"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Revenue bar
    rect_rev = ax.bar(years, rev_total, label="Revenue", alpha=0.7)

    # O&M + degradation bars stacked downward
    rect_om = ax.bar(years, -om_cost, label="O&M cost", alpha=0.7)
    rect_deg = ax.bar(
        years,
        -deg_cost,
        bottom=-om_cost,
        label="Degradation cost",
        alpha=0.7,
    )

    # Cumulative discounted cashflow (after CAPEX)
    ax2 = ax.twinx()
    line_cf, = ax2.plot(
        years,
        cum_disc_cf,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Cum. discounted CF (after CAPEX)",
    )

    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual revenues / costs [EUR]")
    ax2.set_ylabel("Cumulative discounted CF [EUR]")

    title = (
        f"Cashflows and cumulative discounted CF – {scen_name}\n"
        f"CAPEX = {capex:,.0f} EUR, NPV = {npv_project:,.0f} EUR"
    )
    ax.set_title(title)

    # Payback year marker (first year where cum_disc_cf >= 0)
    payback_indices = np.where(cum_disc_cf.values >= 0)[0]
    if len(payback_indices) > 0:
        idx_pb = payback_indices[0]
        year_pb = years.iloc[idx_pb]
        cf_pb = cum_disc_cf.iloc[idx_pb]
        ax2.axvline(year_pb, color="tab:green", linestyle="--", linewidth=1.0)
        ax2.annotate(
            f"Payback ~ Year {int(year_pb)}",
            xy=(year_pb, cf_pb),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=8,
            color="tab:green",
        )

    # Legends
    ax.legend(
        [rect_rev, rect_om, rect_deg, line_cf],
        ["Revenue", "O&M cost", "Degradation cost", "Cum. discounted CF"],
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------
# D. Dispatch & SoC for representative week
# -------------------------------------------------------
def plot_dispatch_sample_week(
    scen_name: str,
    year: int = 1,
    start_hour: int = 0,
    hours: int = 168,
):
    dispatch_path = os.path.join(
        _scenario_dir(scen_name),
        f"dispatch_timeseries_y{year}.csv",
    )
    if not os.path.exists(dispatch_path):
        raise FileNotFoundError(f"{dispatch_path} not found. Run src.run_scenario first.")

    df = pd.read_csv(dispatch_path).sort_values("t")
    df_week = df.iloc[start_hour:start_hour + hours]

    if df_week.empty:
        raise ValueError("Selected week slice is empty. Check start_hour/hours.")

    # Use local hour index for the x-axis of the plot
    h = np.arange(len(df_week))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # Panel 1: DA price
    ax1 = axes[0]
    ax1.plot(h, df_week["price_da_eur_mwh"], label="DA price [€/MWh]")
    ax1.set_ylabel("DA price [€/MWh]")
    ax1.set_title(f"Sample week dispatch – {scen_name}, Year {year} (start_hour={start_hour})")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: SoC
    ax2 = axes[1]
    ax2.plot(h, df_week["soc_mwh"], label="SoC [MWh]", color="tab:blue")
    ax2.set_ylabel("SoC [MWh]")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: charge / discharge power
    ax3 = axes[2]
    ax3.plot(h, df_week["q_ch_mw"], label="q_ch [MW]", linestyle="--")
    ax3.plot(h, df_week["q_dis_mw"], label="q_dis [MW]", linestyle=":")
    ax3.axhline(0.0, color="black", linewidth=0.5)
    ax3.set_ylabel("Power [MW]")
    ax3.set_xlabel("Hour in selected window")
    ax3.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------
# E. Scenario comparison (NPV + Year-1 revenue split)
# -------------------------------------------------------
def plot_npv_comparison(scenarios):
    npvs = []
    for scen in scenarios:
        s = load_kpis_project(scen)
        npvs.append(float(s["npv_eur"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    rects = ax.bar(scenarios, npvs)
    ax.set_ylabel("NPV [EUR]")
    ax.set_title("NPV comparison across scenarios")
    ax.tick_params(axis="x", rotation=10)

    # Add labels on top
    _add_bar_labels(ax, rects, fmt="{:,.0f}")

    fig.tight_layout()
    plt.show()


def plot_year1_revenue_split_comparison(scenarios):
    """
    Year 1 revenue split across scenarios, shown as percentage stacked bars.
    This highlights *composition* of revenue stacking rather than just level.
    """
    labels = ["DA", "aFRR cap up", "aFRR cap down", "aFRR act up", "aFRR act down"]
    values = []

    for scen in scenarios:
        df = load_kpis_per_year(scen)
        row_y1 = df.iloc[0]
        vals = [
            row_y1["rev_da_eur"],
            row_y1.get("rev_afrr_cap_up_eur", 0.0),
            row_y1.get("rev_afrr_cap_down_eur", 0.0),
            row_y1.get("rev_afrr_act_up_eur", 0.0),
            row_y1.get("rev_afrr_act_down_eur", 0.0),
        ]
        values.append(vals)

    values_df = pd.DataFrame(values, index=scenarios, columns=labels)

    # Convert to percentage of total Year-1 revenue
    totals = values_df.sum(axis=1).replace(0.0, np.nan)
    shares = values_df.div(totals, axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(shares.index))

    for col in shares.columns:
        ax.bar(shares.index, shares[col], bottom=bottom, label=col)
        bottom = bottom + shares[col].values

    ax.set_ylabel("Year 1 revenue share [%]")
    ax.set_title("Year 1 revenue split comparison (percentage)")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    print(f"Using MAIN_SCENARIO = {MAIN_SCENARIO}")
    print("Generating plots A–D for main scenario...")

    # A. Revenue split (Year 1 & last year)
    plot_revenue_split_years(MAIN_SCENARIO)

    # B. SoH, revenue, throughput vs year
    plot_soh_revenue_throughput(MAIN_SCENARIO)

    # C. Cashflows and cumulative discounted CF
    plot_cashflows_and_cumulative_npv(MAIN_SCENARIO)

    # D. Dispatch for sample week (Year 1, hours 0–168 by default)
    plot_dispatch_sample_week(MAIN_SCENARIO, year=1, start_hour=0, hours=168)

    # E. Scenario comparison
    if len(SCENARIOS_COMPARE) >= 2:
        print(f"Comparing scenarios: {SCENARIOS_COMPARE}")
        plot_npv_comparison(SCENARIOS_COMPARE)
        plot_year1_revenue_split_comparison(SCENARIOS_COMPARE)
    else:
        print("SCENARIOS_COMPARE has <2 entries; skipping comparison plots.")


if __name__ == "__main__":
    main()
