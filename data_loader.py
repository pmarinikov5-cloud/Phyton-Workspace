from typing import Dict, Any, List, Tuple
import pandas as pd


def _guess_price_column(df: pd.DataFrame, exclude: List[str]) -> str:
    """
    Pick the first column that is not in the exclude list.
    Used for DA price if there is only one such column.
    """
    for col in df.columns:
        if col.lower() not in [e.lower() for e in exclude]:
            return col
    raise ValueError("Could not find a price column (all columns excluded).")


def _guess_up_down_columns(df: pd.DataFrame, exclude: List[str]) -> Tuple[str, str]:
    """
    Try to guess 'up' and 'down' price columns from column names.
    Fallback: take the first two non-excluded columns as (up, down).
    """
    exclude_lower = [e.lower() for e in exclude]
    up_candidates: List[str] = []
    down_candidates: List[str] = []

    for col in df.columns:
        cl = col.lower()
        if cl in exclude_lower:
            continue
        if "up" in cl:
            up_candidates.append(col)
        if "down" in cl:
            down_candidates.append(col)

    if up_candidates and down_candidates:
        return up_candidates[0], down_candidates[0]

    # Fallback: first two non-excluded columns
    others = [col for col in df.columns if col.lower() not in exclude_lower]
    if len(others) < 2:
        raise ValueError("Could not find distinct up/down price columns.")
    return others[0], others[1]


def load_all_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load DA and (optionally) aFRR data from the Excel file specified in config
    and transform into the structures expected by model_bess.build_bess_model().

    If config["enable_afrr"] = false, aFRR sheets are NOT required and the
    returned data will have B=[], hours_in_block={}, price_afrr_*={}
    so the model behaves as DA-only.
    """

    # Flag: DA-only vs DA + aFRR
    enable_afrr = bool(config.get("enable_afrr", True))

    # Accept either key name for the Excel path
    excel_path = config.get("excel_market_file") or config.get("market_data_excel")
    if not excel_path:
        raise KeyError(
            "Provide 'excel_market_file' (preferred) or 'market_data_excel' in the config."
        )

    sheet_da = config.get("sheet_da", "DA")
    sheet_act = config.get("sheet_afrr_act", "aFRR_act")
    sheet_cap = config.get("sheet_afrr_cap", "aFRR_cap")

    # ------------------------------------------------------------------
    # 1) Day-ahead prices (hourly)
    # ------------------------------------------------------------------
    df_da = pd.read_excel(excel_path, sheet_name=sheet_da)

    # Basic sanity: must have a 't' column
    if "t" not in df_da.columns:
        raise ValueError(f"Sheet {sheet_da} must contain a 't' column.")

    df_da = df_da.sort_values("t").reset_index(drop=True)
    T: List[int] = df_da["t"].astype(int).tolist()

    # Guess the DA price column (anything not t/date/time)
    da_price_col = _guess_price_column(df_da, exclude=["t", "date", "time"])
    # Map t -> DA price
    price_da = {
        int(row["t"]): float(row[da_price_col])
        for _, row in df_da.iterrows()
    }

    # ------------------------------------------------------------------
    # 2) Initialise aFRR structures (may stay empty if enable_afrr=False)
    # ------------------------------------------------------------------
    B: List[int] = []
    hours_in_block: Dict[int, List[int]] = {}
    price_afrr_act_up: Dict[int, float] = {}
    price_afrr_act_down: Dict[int, float] = {}
    price_afrr_cap_up: Dict[int, float] = {}
    price_afrr_cap_down: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # 3) aFRR activation & capacity (only if enable_afrr=True)
    # ------------------------------------------------------------------
    if enable_afrr:
        # aFRR activation prices (per hour t)
        df_act = pd.read_excel(excel_path, sheet_name=sheet_act)
        if "t" not in df_act.columns or "b" not in df_act.columns:
            raise ValueError(f"Sheet {sheet_act} must contain 't' and 'b' columns.")
        df_act = df_act.sort_values("t").reset_index(drop=True)

        # t alignment check (optional but good practice)
        t_da_set = set(T)
        t_act_set = set(df_act["t"].astype(int).tolist())
        if t_da_set != t_act_set:
            raise ValueError("Mismatch between DA and aFRR_act t indices.")

        # Guess up/down activation price columns
        act_up_col, act_down_col = _guess_up_down_columns(
            df_act, exclude=["t", "date", "time", "b"]
        )

        price_afrr_act_up = {
            int(row["t"]): float(row[act_up_col])
            for _, row in df_act.iterrows()
        }
        price_afrr_act_down = {
            int(row["t"]): float(row[act_down_col])
            for _, row in df_act.iterrows()
        }

        # Define blocks and hours per block from aFRR_act sheet
        B = sorted(df_act["b"].astype(int).unique().tolist())
        hours_in_block = {
            int(b): df_act.loc[df_act["b"] == b, "t"].astype(int).tolist()
            for b in B
        }

        # aFRR capacity prices (derive per-block average from per-hour data)
        df_cap = pd.read_excel(excel_path, sheet_name=sheet_cap)
        if "t" not in df_cap.columns or "b" not in df_cap.columns:
            raise ValueError(f"Sheet {sheet_cap} must contain 't' and 'b' columns.")
        df_cap = df_cap.sort_values("t").reset_index(drop=True)

        t_cap_set = set(df_cap["t"].astype(int).tolist())
        b_cap_set = set(df_cap["b"].astype(int).tolist())
        if t_cap_set != t_da_set:
            raise ValueError("Mismatch between DA and aFRR_cap t indices.")
        if b_cap_set != set(B):
            raise ValueError("Mismatch between aFRR_act and aFRR_cap block IDs.")

        # Guess up/down capacity price columns
        cap_up_col, cap_down_col = _guess_up_down_columns(
            df_cap, exclude=["t", "date", "time", "b"]
        )

        # Compute average capacity price per block (€/MW/h)
        for b in B:
            sub = df_cap.loc[df_cap["b"] == b]
            if sub.empty:
                price_afrr_cap_up[b] = 0.0
                price_afrr_cap_down[b] = 0.0
            else:
                price_afrr_cap_up[b] = float(sub[cap_up_col].mean())
                price_afrr_cap_down[b] = float(sub[cap_down_col].mean())

    # ------------------------------------------------------------------
    # 4) Grid connection limit
    # ------------------------------------------------------------------
    p_grid_max = config.get("p_grid_max_mw", config.get("battery_power_mw", 30.0))

    # ------------------------------------------------------------------
    # 5) Assemble data dict
    # ------------------------------------------------------------------
    data: Dict[str, Any] = {
        "T": T,
        "B": B,
        "hours_in_block": hours_in_block,
        "price_da": price_da,
        "price_afrr_cap_up": price_afrr_cap_up,
        "price_afrr_cap_down": price_afrr_cap_down,
        "price_afrr_act_up": price_afrr_act_up,
        "price_afrr_act_down": price_afrr_act_down,
        "p_grid_max_mw": p_grid_max,
    }

    # DA-only override: if enable_afrr=False ensure consistent empty structures
    if not enable_afrr:
        data["B"] = []
        data["hours_in_block"] = {}
        data["price_afrr_cap_up"] = {}
        data["price_afrr_cap_down"] = {}
        data["price_afrr_act_up"] = {}
        data["price_afrr_act_down"] = {}

    return data
