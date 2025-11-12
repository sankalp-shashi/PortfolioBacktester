import pandas as pd
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
    return_cols = [c for c in df.columns if c not in ['Holding_Start', 'Holding_End']]
    returns = df[return_cols]

    # Compute cumulative compounded returns
    cumulative = (1 + returns).cumprod() - 1
    cumulative[date_col] = df[date_col]

    if plot:
        plt.figure(figsize=(10, 6))
        for col in return_cols:
            plt.plot(cumulative[date_col], cumulative[col], label=col)
        plt.title("Cumulative Returns Across Holding Periods")
        plt.xlabel("Holding Period End Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()

    return cumulative
