import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from typing import Dict

def load_returns_data(returns_file: str = 'data/prepared_returns.csv') -> pd.DataFrame:
    """
    Load returns data and convert Date column to Timestamp.

    Args:
        returns_file: Path to the CSV file containing returns data

    Returns:
        DataFrame with processed returns data
    """
    try:
        # Read the CSV file
        data = pd.read_csv(returns_file)

        # Convert Date column to Timestamp
        data['Date'] = pd.to_datetime(data['Date'])

        # Sort by date to ensure proper time ordering
        data = data.sort_values('Date').reset_index(drop=True)

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Returns file '{returns_file}' not found.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def run_capm_regression(stock_returns: pd.Series, risk_free_rates: pd.Series,
                       market_excess_returns: pd.Series) -> tuple:
    """
    Run CAPM regression for a single stock.

    Args:
        stock_returns: Stock returns (Ri)
        risk_free_rates: Risk-free rates (in decimals)
        market_excess_returns: Market excess returns (MF in percentage)

    Returns:
        tuple: (alpha, beta, alpha_pvalue, significant)
    """
    # Calculate stock excess returns (Ri - RF) - both in decimals
    stock_excess = stock_returns - risk_free_rates

    # Convert market excess returns from percentage to decimals
    market_excess_decimal = market_excess_returns / 100.0

    # Add constant for alpha (intercept)
    X = sm.add_constant(market_excess_decimal)

    # Remove any NaN values
    valid_mask = ~(np.isnan(stock_excess) | np.isnan(market_excess_decimal))
    stock_excess_clean = stock_excess[valid_mask]
    X_clean = X[valid_mask]

    if len(stock_excess_clean) < 2:
        return 0.0, 0.0, 1.0, False

    # Run regression
    try:
        model = sm.OLS(stock_excess_clean, X_clean)
        results = model.fit()

        alpha = results.params[0]
        beta = results.params[1]
        alpha_pvalue = results.pvalues[0]

        # Test significance at 95% confidence level
        significant = alpha_pvalue < 0.05

        return alpha, beta, alpha_pvalue, significant

    except:
        return 0.0, 0.0, 1.0, False

def form_active_portfolio(holding_start_date: str, holding_end_date: str,
                         returns_file: str = 'data/prepared_returns.csv') -> Dict[str, float]:
    """
    Forms an active portfolio following this logic:
    - Short stocks with significantly positive alpha (overvalued)
    - Long stocks with significantly negative alpha (undervalued)
    - Falls back to market portfolio (NIFTY Index) if no significant alphas found
    """
    # Load and prepare data
    returns_data = load_returns_data(returns_file)

    # Convert input dates to Timestamp
    start_dt = pd.to_datetime(holding_start_date)
    end_dt = pd.to_datetime(holding_end_date)

    # Extract formation period data
    mask = (returns_data['Date'] >= start_dt) & (returns_data['Date'] <= end_dt)
    formation_data = returns_data[mask].copy()
    if formation_data.empty:
        raise ValueError(f"No data found for formation period {holding_start_date} to {holding_end_date}")

    # Identify stock columns (exclude Date, RF, MF, and 'NIFTY Index')
    exclude_cols = ['Date', 'RF', 'MF', 'NIFTY Index']
    stock_columns = [col for col in formation_data.columns if col not in exclude_cols]
    if not stock_columns:
        raise ValueError("No stock columns found in the data")

    # Get risk-free rate and market excess return columns
    if 'RF' not in formation_data.columns:
        raise ValueError("Risk-free rate column 'RF' not found in data")
    if 'MF' not in formation_data.columns:
        raise ValueError("Market excess return column 'MF' not found in data")

    risk_free_rates = formation_data['RF']  # decimals
    market_excess_returns = formation_data['MF']  # in percentage

    # Run CAPM regression for each stock
    significant_stocks = {}
    for stock in stock_columns:
        stock_returns = formation_data[stock]  # decimals
        alpha, beta, alpha_pvalue, significant = run_capm_regression(
            stock_returns, risk_free_rates, market_excess_returns
        )
        if significant:
            significant_stocks[stock] = {'alpha': alpha, 'beta': beta, 'pvalue': alpha_pvalue}

    # Fallback: no significant alphas
    if not significant_stocks:
        print("[INFO] No stocks with significant alpha found. Returning market portfolio instead of active.")
        return {'NIFTY Index': 1.0}

    # --- CAPM believer logic: go LONG negative-alpha, SHORT positive-alpha stocks ---
    alphas = np.array([info['alpha'] for info in significant_stocks.values()])
    stock_names = list(significant_stocks.keys())

    # Flip sign of alphas (so weights ∝ -alpha)
    raw_weights = -alphas

    # Normalize by gross exposure (sum of absolute weights = 1)
    gross_exposure = np.sum(np.abs(raw_weights))
    if gross_exposure == 0:
        print("[WARN] All alphas are zero. Returning market portfolio.")
        return {'NIFTY Index': 1.0}

    weights = raw_weights / gross_exposure

    # Build portfolio dictionary
    portfolio = {stock: weight for stock, weight in zip(stock_names, weights)}

    return portfolio
