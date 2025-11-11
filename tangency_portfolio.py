import pandas as pd
import numpy as np

def tangency_portfolio( start_date, end_date):
    """
    Compute Tangency (Mean-Variance Efficient) Portfolio Weights.
    
    Parameters:
        file_path (str): Path to CSV file containing daily returns.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        dict: Stock names as keys and Tangency weights as values.
    """
    
    # Step 1: Read CSV
    file_path = "data/prepared_returns.csv"
    df = pd.read_csv(file_path)
    
    # Step 2: Parse dates and set index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Step 3: Filter by date range
    df = df.loc[start_date:end_date]
    
    # Step 4: Extract and drop unwanted columns
    rf_series = df['RF'] if 'RF' in df.columns else pd.Series(0, index=df.index)
    df_assets = df.drop(columns=['RF', 'MF', 'NIFTY'], errors='ignore')
    
    # Step 5: Drop missing values
    df_assets = df_assets.dropna()
    
    # Step 6: Compute mean returns and covariance matrix
    mean_returns = df_assets.mean().values      # μ vector
    cov_matrix = df_assets.cov().values         # Σ matrix
    
    # Step 7: Compute average risk-free rate over the period
    rf = rf_series.mean()                       # scalar
    
    # Step 8: Compute excess returns (μ - r_f)
    excess_returns = mean_returns - rf
    
    # Step 9: Compute inverse of covariance
    inv_cov = np.linalg.inv(cov_matrix)
    
    # Step 10: Compute Tangency weights
    numerator = inv_cov @ excess_returns
    denominator = np.sum(numerator)
    weights = numerator / denominator
    
    # Step 11: Convert to dictionary
    weights_dict = dict(zip(df_assets.columns, weights))
    
    return weights_dict


# Example usage:
weights = tangency_portfolio("2015-01-01", "2015-06-30")
print(weights)
