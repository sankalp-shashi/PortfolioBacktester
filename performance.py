import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_returns(csv_path, date_col="Holding_Start", plot=True):
    """
    Reads a CSV of rolling holding-period returns and computes cumulative compounded returns.

    Expected columns:
        - 'Holding_Start', 'Holding_End' (optional for plotting)
        - One or more strategy columns (each containing 3-month simple returns in decimals)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing rolling returns.
    date_col : str
        Column name representing the holding-period end date (default: 'Holding_End').
    plot : bool
        Whether to display a cumulative return plot.

    Returns
    -------
    cumulative : pd.DataFrame
        DataFrame containing cumulative compounded returns for each strategy.
    """
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract only return columns (exclude date columns)
    return_cols = [c for c in df.columns if c not in ['Holding_Start', 'Holding_End', 'RF']]
    returns = df[return_cols]

    # Compute cumulative compounded returns
    cumulative = (1 + returns).cumprod() - 1
    cumulative[date_col] = df[date_col]

    if plot:
        plt.figure(figsize=(10, 6))
        for col in return_cols:
            plt.plot(cumulative[date_col], cumulative[col], label=col)
        plt.title("Cumulative Returns Across Holding Periods")
        plt.xlabel("Holding Period Start Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()

    return cumulative

def compute_performance_table(results_file: str = 'data/returns_6M_3M.csv',
                              trading_days_per_year: int = 252,
                              trading_days_per_window: int = 63) -> pd.DataFrame:
    """
    Compute annualized performance metrics for each portfolio strategy
    (Mean Return, Std Dev, Sharpe, Information Ratio).

    Uses realized per-window risk-free (RF) and market (NIFTY Index) returns.

    Parameters
    ----------
    results_file : str
        Path to CSV file containing columns:
        Holding_Start, Holding_End, EW, GMV, TNG, ACTIVE, RF, NIFTY Index
    trading_days_per_year : int
        Assumed number of trading days in a year (default: 252)
    trading_days_per_window : int
        Trading days per holding window (default: 63 ≈ 3 months)

    Returns
    -------
    pd.DataFrame
        Performance table with annualized metrics per strategy.
    """
    df = pd.read_csv(results_file)
    # Identify strategy columns (exclude non-strategy ones)
    exclude_cols = ['Holding_Start', 'Holding_End', 'RF']
    strategy_cols = [c for c in df.columns if c not in exclude_cols]

    # Annualization factors
    n_periods_per_year = trading_days_per_year / trading_days_per_window
    sqrt_factor = np.sqrt(n_periods_per_year)

    perf = {}

    # Market reference
    market_col = 'NIFTY Index'
    market_ret = df[market_col].dropna()

    for strat in strategy_cols:
        r = df[strat].dropna()
        rf = df.loc[r.index, 'RF']  # align RF to same rows

        # Compute excess returns per window
        excess_returns = r - rf

        # Mean and std of 3-month returns
        mean_q = np.mean(r)
        std_q = np.std(r, ddof=1)
        mean_excess_q = np.mean(excess_returns)
        std_excess_q = np.std(excess_returns, ddof=1)

        # Annualized metrics
        mean_ann = (1 + mean_q) ** n_periods_per_year - 1
        std_ann = std_q * sqrt_factor
        mean_excess_ann = (1 + mean_excess_q) ** n_periods_per_year - 1
        std_excess_ann = std_excess_q * sqrt_factor

        # Sharpe ratio (annualized, using realized RF)
        sharpe = mean_excess_ann / std_excess_ann if std_excess_ann > 0 else np.nan

        perf[strat] = {
            "Mean_Annual": mean_ann,
            "Std_Annual": std_ann,
            "Sharpe": sharpe
        }

    # Compute Information Ratios relative to NIFTY Index
    mean_market_ann = (1 + np.mean(market_ret)) ** n_periods_per_year - 1

    for strat in strategy_cols:
        if strat == market_col:
            perf[strat]["Info_Ratio"] = np.nan
            continue
        diff = df[strat] - df[market_col]
        diff_std_ann = np.std(diff, ddof=1) * sqrt_factor
        info = (perf[strat]["Mean_Annual"] - mean_market_ann) / diff_std_ann if diff_std_ann > 0 else np.nan
        perf[strat]["Info_Ratio"] = info

    results_df = pd.DataFrame(perf).T
    return results_df


def plot_var_vs_realized(strategies, base_path='data/', output_file='data/VaR_vs_Realized_All.png'):
    """
    Create a 2x2 grid of subplots for each strategy showing realized returns vs 99% VaR.
    Violation count is shown in the title (no highlighted points).
    Saves a single PNG file and displays it.
    """
    n = len(strategies)
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for i, strat in enumerate(strategies):
        if i >= len(axes):
            break  # safety for fewer than 4 strategies

        ax = axes[i]
        file_path = f"{base_path}/var_results_{strat}.csv"

        try:
            df = pd.read_csv(file_path, parse_dates=["Window_Start", "Window_End"])
        except FileNotFoundError:
            print(f"[WARN] File not found: {file_path}")
            ax.set_visible(False)
            continue

        # Clean and sort
        df["VaR_99"] = pd.to_numeric(df["VaR_99"], errors="coerce")
        df["Realized_Return"] = pd.to_numeric(df["Realized_Return"], errors="coerce")
        df["Violation"] = df["Violation"].astype(str).str.lower() == "true"
        df = df.sort_values("Window_Start").reset_index(drop=True)

        # Count violations
        violation_count = df["Violation"].sum()

        # Plot
        ax.plot(df["Window_Start"], df["Realized_Return"], label="Realized Return", color="tab:blue")
        ax.plot(df["Window_Start"], df["VaR_99"], label="99% VaR (Negative)", color="tab:red", linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f"{strat} Portfolio  |  Violations: {violation_count}")
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.3)

    # Hide any empty axes if fewer than 4 strategies
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared x-label
    for ax in axes[-2:]:
        ax.set_xlabel("Window Start Date")

    # Common legend (top center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Save + show
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Saved combined VaR vs. Realized plot to {output_file}")
    plt.show()
