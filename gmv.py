import pandas as pd
import numpy as np

def global_min_variance_portfolio(start_date, end_date):
    """
    Compute Global Minimum Variance (GMV) Portfolio Weights.
    
    Parameters:
        file_path (str): Path to the CSV file containing daily returns.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        dict: Stock names as keys and GMV weights as values.
    """
    
    # Step 1: Read CSV
    
    df = pd.read_csv("data/prepared_returns.csv")
    
    # Step 2: Parse dates and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Step 3: Filter by date range
    df = df.loc[start_date:end_date]
    
    # Step 4: Drop non-asset columns
    df_assets = df.drop(columns=['RF', 'MF', 'NIFTY Index'], errors='ignore')
    
    # # Step 5: Drop rows with missing values (if any)
    # df_assets = df_assets.dropna()
    
    # Step 6: Compute covariance matrix
    cov_matrix = df_assets.cov().values
    
    # Step 7: Compute inverse of covariance matrix
    inv_cov = np.linalg.inv(cov_matrix)
    
    # Step 8: Create a vector of ones
    ones = np.ones(len(df_assets.columns))
    
    # Step 9: Compute GMV weights
    weights = inv_cov @ ones
    weights = weights / (ones.T @ inv_cov @ ones)
    
    # Step 10: Return as dictionary
    weights_dict = dict(zip(df_assets.columns, weights))
    
    return weights_dict


