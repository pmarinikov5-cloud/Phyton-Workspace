Model Description – BESS DA + aFRR Co-Optimisation
*ADD .VENV ENVIRONMENT 
This section documents the core optimisation model implemented in src/model_bess.py. The model is a (mixed-integer) linear program that co-optimises Day-Ahead (DA) arbitrage and automatic Frequency Restoration Reserve (aFRR) participation for a utility-scale BESS over one year, given exogenous price series and an exogenous SoH (usable energy) for that year.

The model is solved once per project year; multi-year chaining, SoH path and financial aggregation are handled externally in run_scenario.py and financials.py.

Purpose and scope

The model decides, for every hour of the year and every aFRR block:

– How the battery charges and discharges (power and SoC trajectory)
– What DA net position is scheduled at the point of interconnection (POI)
– What aFRR capacity is offered (up or down) in each 4-hour block
– How much converter headroom is reserved to fulfil aFRR commitments

Subject to:

– Battery power and energy limits
– SoC bounds (shrinking over time via SoH)
– Converter rating
– ATR85 non-firm grid access constraint on DA net exchange
– aFRR direction exclusivity within each block (up OR down, not both)
– SoC feasibility for expected aFRR activation (energy headroom/footroom)
– Avoiding simultaneous charge/discharge and simultaneous import/export

The objective is to maximise annual gross margin (DA + aFRR capacity + expected aFRR activation revenue) minus:

– Throughput-based degradation proxy (optional)
– Block-end SoC deviation penalties (soft anchoring)

Indices and sets

T – Set of hourly time steps in the year, e.g. T = {0,…,8759}
B – Set of aFRR blocks (e.g. 4-hour blocks), indexed so that each block b ∈ B maps to a set of hours hours_in_block[b] ⊆ T

A mapping t → b is built internally so each hour is linked to its aFRR block for capacity/activation handling.

Key input parameters

From the configuration:

– P_batt [MW] – Converter/battery power rating
– E_year [MWh] – Usable energy for that year (E0 × SoH_year), passed via config
– soc_min_frac, soc_max_frac [–] – Min/max SoC as a fraction of E_year
– soc_init_frac [–] – Initial SoC fraction at start of year
– η_ch, η_dis [–] – Charge/discharge efficiencies
– α_ATR [–] – ATR85 availability factor on grid connection
– activation_ratio_up, activation_ratio_down [–] – Global expected aFRR activation ratios (ρ↑, ρ↓)
– λ_throughput [€/MWh] – Weight for throughput penalty (optional)
– λ_soc_block [€/MWh] – Weight for block-end SoC deviation penalty (optional)

From the data loader (data_loader.py):

– price_da[t] [€/MWh] – Day-Ahead price per hour t
– price_afrr_cap_up[b] [€/MW/h] – aFRR capacity price up for block b
– price_afrr_cap_down[b] [€/MW/h] – aFRR capacity price down for block b
– price_afrr_act_up[t] [€/MWh] – aFRR activation price up (energy)
– price_afrr_act_down[t] [€/MWh] – aFRR activation price down (energy)
– hours_in_block[b] – Mapping from block b to its 4 constituent hours

Decision variables

Hourly power and SoC:

– q_ch[t] ≥ 0 [MW] – Charging power at the converter (AC-side)
– q_dis[t] ≥ 0 [MW] – Discharging power at the converter (AC-side)
– soc[t] [MWh] – State of charge, bounded between soc_min_mwh and soc_max_mwh

Grid exchange at POI:

– g_imp[t] ≥ 0 [MW] – Grid import at the POI
– g_exp[t] ≥ 0 [MW] – Grid export at the POI
– p_da_net[t] [MW] – Signed DA net position at POI, defined as p_da_net[t] = g_exp[t] − g_imp[t]

aFRR block-level variables:

– u_b_up[b] ≥ 0 [MW] – aFRR upward capacity offered in block b
– u_b_down[b] ≥ 0 [MW] – aFRR downward capacity offered in block b
– z_b_dir[b] ∈ {0,1} – Binary direction flag: 1 = block is configured as “up” (u_b_up can be > 0), 0 = “down” (u_b_down can be > 0)

Converter reservation for aFRR (per hour):

– p_afrr_req[t] ≥ 0 [MW] – Required converter headroom in hour t used by aFRR capacity; linked to u_b_up/u_b_down of the corresponding block

Block-end SoC anchoring:

– z_block_dev[b] ≥ 0 [MWh] – Slack variable capturing absolute deviation of block-end SoC from a target reference (typically 50% of E_year)

Charge/discharge and import/export mode (binaries – conceptual):

– Binary “mode” variables (per hour) are introduced to enforce that the model does not use both directions on the same interface in the same hour:
• charging vs discharging (q_ch vs q_dis)
• importing vs exporting (g_imp vs g_exp)
These are implemented via standard big-M style constraints tying the continuous variables to the binary flags, so that at most one direction in each pair can be active at significant magnitude in a given hour.

Core constraints

5.1 SoC dynamics

For each hour t > t0:

soc[t] = soc[t−1] + η_ch · q_ch[t] · Δt − q_dis[t] · Δt / η_dis

The initial SoC is fixed to:

soc[t0] = soc_init_frac · E_year

SoC bounds (energy window):

soc_min = soc_min_frac · E_year ≤ soc[t] ≤ soc_max_frac · E_year = soc_max

SoC bounds shrink with SoH because E_year itself is reduced each year.

5.2 Power balance at the POI

For each hour t:

g_imp[t] − g_exp[t] = q_dis[t] − q_ch[t]

This ensures that all power flows through the converter; there is no parallel path around the battery.

The DA schedule at the POI is defined by:

p_da_net[t] = g_exp[t] − g_imp[t]

Positive p_da_net[t] means net export (discharging into the grid), negative p_da_net[t] means net import (charging from the grid).

5.3 Converter limits and ATR85

Grid connection (ATR85 as physical derating):

|p_da_net[t]| ≤ α_ATR · P_batt

This is implemented as two linear inequalities:

p_da_net[t] ≤ p_grid_max
−p_da_net[t] ≤ p_grid_max

with p_grid_max = α_ATR · P_batt.

Individual bounds on import/export:

g_imp[t] ≤ p_grid_max
g_exp[t] ≤ p_grid_max

Converter limit for the combination of DA schedule and aFRR requirement:

|p_da_net[t]| + p_afrr_req[t] ≤ P_batt

Linearised:

p_da_net[t] + p_afrr_req[t] ≤ P_batt
−p_da_net[t] + p_afrr_req[t] ≤ P_batt

This ensures the converter can always accomodate the DA baseline plus any aFRR activation up to the committed capacity.

5.4 Battery power bounds and (approximate) mutual exclusivity

For each hour t:

0 ≤ q_ch[t] ≤ P_batt
0 ≤ q_dis[t] ≤ P_batt
q_ch[t] + q_dis[t] ≤ P_batt

The last inequality combined with binary mode variables prevents the model from economically exploiting artificial simultaneous charge and discharge; it enforces “one direction at a time” behaviour up to a small numerical tolerance.

5.5 aFRR direction exclusivity and converter reservation

For each block b:

u_b_up[b] ≤ z_b_dir[b] · P_batt
u_b_down[b] ≤ (1 − z_b_dir[b]) · P_batt

So each block is exclusively up or down; not both.

For each hour t in block b:

p_afrr_req[t] ≥ u_b_up[b]
p_afrr_req[t] ≥ u_b_down[b]
p_afrr_req[t] ≤ u_b_up[b] + u_b_down[b]
p_afrr_req[t] ≤ P_batt

Given the directional binary, at most one of u_b_up, u_b_down is non-zero, so p_afrr_req[t] effectively becomes that active capacity at block level. This is what enters the converter limit constraint together with p_da_net[t].

5.6 SoC feasibility for expected aFRR activation

For each hour t in block b:

Upward activation headroom:

soc[t] ≥ ρ_up · u_b_up[b] · Δt

Downward activation footroom:

E_year − soc[t] ≥ ρ_down · u_b_down[b] · Δt

These constraints guarantee that, given the expected activation ratios (ρ_up, ρ_down), the battery has enough energy to deliver an upward energy volume or enough empty space to absorb a downward energy volume without violating SoC bounds.

5.7 Block-end SoC anchoring

Let t_end be the last hour of block b and soc_ref be the mid-SoC target (typically 0.5 · E_year). Then:

z_block_dev[b] ≥ soc[t_end] − soc_ref
z_block_dev[b] ≥ −(soc[t_end] − soc_ref)

z_block_dev[b] is the absolute deviation of SoC at the end of the block from the reference. This slack is penalised in the objective to discourage uncontrolled SoC drift across the year while still allowing the optimiser some flexibility if economic conditions justify deviating from the anchor.

Objective function

The objective maximises:

Total annual gross margin =
(1) DA arbitrage revenue

(2) aFRR capacity revenue

(3) Expected aFRR activation revenue
− (4) Throughput penalty (optional, proxy for degradation)
− (5) Block-end SoC deviation penalty (optional regularisation)

Formal structure:

Day-Ahead arbitrage:

Σ_t price_da[t] · p_da_net[t] · Δt

aFRR capacity revenue:

Σ_b ( price_afrr_cap_up[b] · u_b_up[b] + price_afrr_cap_down[b] · u_b_down[b] ) · block_duration_b

where block_duration_b = |hours_in_block[b]| · Δt.

Expected aFRR activation revenue (using global activation ratios):

Σ_t ( price_afrr_act_up[t] · ρ_up · u_b_up[b(t)] + price_afrr_act_down[t] · ρ_down · u_b_down[b(t)] ) · Δt

Here b(t) is the block that contains hour t.

Throughput penalty (if λ_throughput ≠ 0):

λ_throughput · Σ_t 0.5 · (q_ch[t] + q_dis[t]) · Δt

This term is subtracted from revenue; it discourages excessive cycling and serves as a simple proxy for degradation cost at the model level.

SoC anchoring penalty (if λ_soc_block ≠ 0):

λ_soc_block · Σ_b z_block_dev[b]

Also subtracted from revenue; it discourages the optimiser from pushing SoC to extremes at block boundaries whenever that has low economic justification.

Modelling assumptions and limitations

– Activation ratios are global constants (ρ_up, ρ_down), not time-series; actual real-world variability of activation is simplified to a deterministic expectation.
– The aFRR capacity is offered per 4-hour block with a fixed direction (up or down); mixed direction blocks are not allowed.
– ATR85 is implemented as a physical derating (limit on |p_da_net|) rather than a stochastic availability pattern or explicit curtailment schedule.
– Degradation is not modelled via internal state-of-health dynamics; instead, SoH is exogenous per year (E_year = E0 · SoH_y) and any intra-year degradation is represented via a throughput penalty and the external financial module.
– Binary mode variables are used to avoid pathological simultaneous import/export or charge/discharge behaviour, but small numerical “leakage” may still appear at the level of a few kilowatts/hours due to solver tolerances; these are treated as negligible.

How this plugs into the rest of the pipeline

run_scenario.py calls build_bess_model(...) once per project year, with E_year already scaled by the exogenous SoH path. After solving, it:

– Reads all decision variables into a dispatch DataFrame (via postprocess.build_dispatch_dataframe)
– Computes annual revenues and throughput (compute_revenue_and_throughput)
– Aggregates cashflows, CAPEX, OPEX and degradation costs (compute_financials_multi_year)
– Exports dispatch_timeseries_y*.csv, kpis_per_year.csv and kpis_project.csv for further analysis, plotting and inclusion in the thesis.


This means model_bess.py is a pure “single-year engine”: it only knows about one year’s market data and one year’s usable energy, and it focuses solely on the physical and market-rule consistency of DA + aFRR co-optimisation under ATR85.
