import pandas as pd

def get_equal_weighted_portfolio_weights():
    # 1. Read CSV
    df = pd.read_csv("data/prepared_returns.csv")
    
    # 2. Ensure 'Date' column exists
    if 'Date' not in df.columns:
        raise ValueError("CSV must have a 'Date' column")
    
    # 3. Convert and set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 4. Drop unwanted columns before computing asset columns
    df = df.drop(columns=['MF'], errors='ignore')  # 'ignore' avoids error if MF doesn't exist
    
    # 5. Asset columns (all remaining after dropping Date and MF)
    asset_cols = df.columns.tolist()
    
    # 6. Compute equal weights
    n_assets = len(asset_cols)
    weights = pd.Series([1/n_assets]*n_assets, index=asset_cols)
    
    return weights

weights = get_equal_weighted_portfolio_weights()
print(weights)
