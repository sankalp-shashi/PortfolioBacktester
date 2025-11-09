import json
import pandas as pd
from pathlib import Path

def init_window_config(start_date: str, formation_months: int, holding_months: int, config_path: str = "config.json"):
    """
    Initializes the rolling window configuration file.
    """
    config = {
        "formation_start": start_date,
        "formation_months": formation_months,
        "holding_months": holding_months
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Initialized config at {config_path}")
    return config


def get_next_window(df, config_path="config.json"):
    """
    Returns the next rolling (formation, holding) window.
    Moves forward by holding period each call.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError("Missing config.json. Initialize first.")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    all_dates = df["Date"]
    formation_start = pd.to_datetime(cfg["formation_start"])
    formation_months = int(cfg["formation_months"])
    holding_months = int(cfg["holding_months"])


    # Compute end boundaries
    formation_end = formation_start + pd.DateOffset(months=formation_months)
    holding_end = formation_end + pd.DateOffset(months=holding_months)

    # Snap ends to nearest valid trading days
    f_start_idx = all_dates.searchsorted(formation_start, side="right") - 1
    f_end_idx = all_dates.searchsorted(formation_end, side="right") - 1
    h_end_idx = all_dates.searchsorted(holding_end, side="right") - 1

    if f_end_idx == h_end_idx or h_end_idx >= len(all_dates) or f_end_idx <= f_start_idx:
        print("[INFO] Reached end of dataset.")
        return None

    formation_end = all_dates.iloc[f_end_idx]
    holding_end = all_dates.iloc[h_end_idx]

    # Prepare output
    next_window = (formation_start, formation_end, holding_end)
    print(f"[INFO] Window: Formation {formation_start.date()} → {formation_end.date()}, Holding → {holding_end.date()}")

    # Advance formation start by holding_months for next call
    next_start = formation_start + pd.DateOffset(months=holding_months)
    cfg["formation_start"] = str(next_start.date())
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)

    return next_window
