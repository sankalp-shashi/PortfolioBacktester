import os
import pandas as pd

def merge_data(stocks_file: str, factors_file: str, output_dir: str = "data") -> pd.DataFrame:
    stocks_df = pd.read_csv(stocks_file, parse_dates=["Dates"])
    factors_df = pd.read_csv(factors_file, parse_dates=["Date"])

    merged_df = pd.merge(
            stocks_df,
            factors_df,
            left_on="Dates",
            right_on="Date",
            how="inner"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_data.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"[INFO] Merged data saved to: {output_path}")
    print(f"[INFO] Final dataset shape: {merged_df.shape}")
    return merged_df
