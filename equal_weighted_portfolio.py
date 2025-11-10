import pandas as pd

def get_equal_weighted_portfolio_weights():

    # 1. Read CSV
    df = pd.read_csv("data/prepared_returns.csv")
    
    # 2. Ensure 'Date' column exists
    if 'Date' not in df.columns:
        raise ValueError("CSV must have a 'Date' column")
    
    # 3. Set Date column as index (optional, for convenience)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 4. Extract asset columns (all except Date)
    asset_cols = df.columns.tolist()
    
    # 5. Number of assets
    n_assets = len(asset_cols) - 2
    
    # 6. Equal weights (each gets 1/n)
    weights = pd.Series([1/n_assets]*n_assets, index=asset_cols)
    
    return weights
