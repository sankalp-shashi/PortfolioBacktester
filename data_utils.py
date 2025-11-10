import os
import pandas as pd


def merge_data(stocks_file: str, factors_file: str, output_dir="data") -> pd.DataFrame:
    stocks_df = pd.read_csv(stocks_file, parse_dates=["Dates"], dayfirst=True)
    factors_df = pd.read_csv(factors_file, parse_dates=["Date"], dayfirst=True)

    # rename the stocks_df to have the expected date column name
    stocks_df = stocks_df.rename(columns={"Dates": "Date"})
    merged_df = pd.merge(
            stocks_df,
            factors_df,
            on="Date",
            how="inner"
        )

    # Sort by date before saving
    merged_df = merged_df.sort_values('Date')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_data.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"[INFO] Merged data saved to: {output_path}")
    print(f"[INFO] Final dataset shape: {merged_df.shape}")
    return merged_df

def build_static_stock_universe(df: pd.DataFrame) -> pd.DataFrame:
    # stock_cols = [
    #     c for c in df.columns
    #     if c not in ["Date","MF","RF"] and "NIFTY" not in c.upper()
    # ]

    df_clean = df.dropna(axis=1, how="any").copy()
    print(f"[INFO] Dropped incomplete stocks. Remaining: {len(df.columns)} columns.")
    print(f"[INFO] Rows after cleaning: {df_clean.shape[0]}")
    return df_clean


def convert_prices_to_returns(df: pd.DataFrame, output_dir: str = "data") -> pd.DataFrame:
    """
    Converts all price columns (stocks + NIFTY Index) to daily simple returns.
    MF and RF columns are kept as-is.

    Saves the resulting DataFrame to data/prepared_returns.csv.
    """
    price_cols = [c for c in df.columns if c not in ["Date", "MF", "RF"]]
    returns_df = df[["Date"]].copy()

    for col in price_cols:
        returns_df[col] = df[col].pct_change()

    returns_df["MF"] = df["MF"]
    returns_df["RF"] = df["RF"]
    returns_df = returns_df.dropna().reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prepared_returns.csv")
    returns_df = returns_df.sort_values("Date").reset_index(drop=True)
    returns_df.to_csv(output_path, index=False)

    print(f"[INFO] Converted prices to daily returns.")
    print(f"[INFO] Saved prepared returns to: {output_path}")
    print(f"[INFO] Shape: {returns_df.shape}")
    return returns_df

