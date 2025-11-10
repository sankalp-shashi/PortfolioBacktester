import json
import pandas as pd
import numpy as np
import os
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


def compute_holding_return(prices_df, weights_dict, holding_start, holding_end):
    """
    Compute holding-period return for each strategy using start & end prices.
    Assumes no rebalancing within window.
    """
    start_prices = prices_df.loc[prices_df["Date"] == holding_start]
    end_prices = prices_df.loc[prices_df["Date"] == holding_end]
    if start_prices.empty or end_prices.empty:
        print(f"[WARN] Missing prices for {holding_start} or {holding_end}")
        return None

    results = {}
    for strat, weights in weights_dict.items():
        w = pd.Series(weights)
        valid_stocks = [c for c in w.index if c in prices_df.columns]

        # Portfolio value at start and end
        v_start = (start_prices[valid_stocks].values.flatten() * w[valid_stocks]).sum()
        v_end = (end_prices[valid_stocks].values.flatten() * w[valid_stocks]).sum()

        results[strat] = (v_end / v_start) - 1

    return results


def calculate_var_for_strategies(sample_returns, quantile_level=0.01):
    """
    Given a list of dicts (each containing strategy returns for one L-day window),
    compute 99% Historical VaR for each strategy.
    """
    strat_names = sample_returns[0].keys()
    var_dict = {}

    for strat in strat_names:
        # Extract that strategy's returns across all L-day samples
        strat_sample = [r[strat] for r in sample_returns if r[strat] is not None]
        if len(strat_sample) == 0:
            var_dict[strat] = np.nan
            continue

        q = np.percentile(strat_sample, quantile_level * 100)
        var_dict[strat] = -q  # convert to positive loss number

    return var_dict


def compute_historical_var(prices_df, weights_dict, formation_start, formation_end, holding_length):
    """
    Uses compute_holding_return() repeatedly on overlapping L-day windows within formation period
    to get the 1% quantile (99% VaR).
    """
    all_dates = prices_df['Date'].sort_values().unique()
    f_mask = (prices_df['Date'] >= formation_start) & (prices_df['Date'] <= formation_end)
    f_dates = all_dates[np.where(f_mask)[0]]

    L = holding_length
    sample_returns = []

    for i in range(0, len(f_dates) - L):
        r_start = f_dates[i]
        r_end = f_dates[i + L - 1]
        r = compute_holding_return(prices_df, weights_dict, r_start, r_end)
        if r is not None:
            sample_returns.append(r)

    if len(sample_returns) == 0:
        return None

    return calculate_var_for_strategies(sample_returns)


def rolling_backtest(
    prices_path="data/prices.csv",
    weights_dict=None,
    formation_months=6,
    holding_months=3,
    start_date="2009-01-01",
    output_dir="data"
):
    """
    Runs a rolling backtest across the full dataset using fixed-weight portfolios.

    Parameters
    ----------
    prices_path : str
        Path to the cleaned daily prices CSV.
    weights_dict : dict
        Nested dictionary of strategy -> stock -> weight.
        Assumed constant for all windows in this simple version.
    formation_months, holding_months : int
        Window lengths (in months).
    start_date : str
        Start date for first formation window.
    output_dir : str
        Directory to store results and config.
    """

    os.makedirs(output_dir, exist_ok=True)
    config_path = Path(output_dir) / "config.json"

    # --- 1. Initialize config file ---
    config = {
        "formation_start": start_date,
        "formation_months": formation_months,
        "holding_months": holding_months
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Initialized config with start={start_date}")

    # --- 2. Load cleaned prices ---
    prices_df = pd.read_csv(prices_path)
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.tz_localize(None)
    prices_df = prices_df.sort_values("Date").reset_index(drop=True)

    # --- 3. Prepare output file ---
    output_file = Path(output_dir) / f"returns_{formation_months}M_{holding_months}M.csv"
    if output_file.exists():
        output_file.unlink()  # start fresh
    print(f"[INFO] Output file: {output_file}")

    # --- 4. Rolling loop ---
    iteration = 1
    while True:
        window = get_next_window(prices_df, config_path)
        if window is None:
            print("[INFO] Backtest complete.")
            break

        formation_start, formation_end, holding_end = window
        holding_start = formation_end  # holding starts right after formation

        print(f"\n[INFO] Iteration {iteration}: Holding {holding_start.date()} → {holding_end.date()}")


        # Compute the historical 99% VaR for all strategies
        var_info = compute_historical_var(
            prices_df,
            weights_dict,
            formation_start,
            formation_end,
            holding_length=len(
                prices_df.loc[
                    (prices_df['Date'] > holding_start) & (prices_df['Date'] <= holding_end),
                    'Date'
                ].sort_values().to_numpy()
            )
        )
        if var_info is None:
            continue

        # Compute holding-period return for all strategies
        results = compute_holding_return(
            prices_df,
            weights_dict,
            holding_start,
            holding_end
        )


        # Compare VaR values with realized holding period returns
        for strat_name, VaR_99 in var_info.items():
            realized_return = results.get(strat_name, None)
            if realized_return is None or np.isnan(VaR_99):
                continue

            violation = realized_return < -VaR_99

            # prepare output row
            record = {
                "Window_Start": formation_start,
                "Window_End": holding_end,
                "VaR_99": VaR_99,
                "Realized_Return": realized_return,
                "Violation": violation
            }


            out_dir = Path("data")
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"var_results_{strat_name}.csv"
            pd.DataFrame([record]).to_csv(
                out_file,
                mode="a",
                header=not out_file.exists(),
                index=False
            )

            print(f"[INFO] {strat_name}: VaR={VaR_99:.4f}, Return={realized_return:.4f}, Violation={violation}")


        # Append to results CSV
        if results:
            df_out = pd.DataFrame([{
                "Holding_Start": holding_start.date(),
                "Holding_End": holding_end.date(),
                **results
            }])
            header = not output_file.exists()
            df_out.to_csv(output_file, mode="a", index=False, header=header)

        iteration += 1
