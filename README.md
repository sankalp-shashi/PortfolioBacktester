# TODO:
~~1. `Data Cleanup (static and dynamic)`~~
~~2. `Implement Portfolio Builders for each strategy that take in data for the formation window`~~
~~3. `Implement the Portfolio Backtester that takes in data for the holding window and records returns for each portfolio`~~
~~4. `Implement the rolling window function that provides data to the builders and backtester`~~
~~5. `Implement the performance metric calculators that read from the recorded returns`~~
~~6. `VaR backtesting`~~
7. `Compile everything into a nice pdf to submit`
8. `Figure out where to store different results and under what names, everything is dumped into data for now`
~~9. `Verify code correctness`~~

# High Level Design:
PortfolioBacktester/
│
├── data_utils.py
├── backtester.py
├── performance.py
└── main.py


### `data_utils.py`
Functions for loading, cleaning, and aligning stock and factor data. Handles static or dynamic universe filtering.

### `backtester.py`
Single function that runs the rolling formation/holding backtest, calls strategy builders, and records realized returns.

### `performance.py`
Standalone functions to compute evaluation metrics (e.g., Annualized Return, Volatility, Sharpe Ratio, Info Ratio).

### `main.py`
Project entry point that orchestrates the workflow, loads data, defines strategies, runs the backtest, and summarizes results.



