import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os


def get_stock_data(symbol, start_date, end_date):
    result = yf.download(symbol, start=start_date, end=end_date)
    result.to_csv(f'data/{symbol}.csv')
    return result


# Define trading cost and tax rate
trading_cost = 0.001  # 0.1% of the trade value
tax_rate = 0.15  # 15% of the realized profit


def calculate_strategy_returns(df, short_window, long_window, trading_cost, tax_rate):
    df['short_mavg'] = df['Close'].rolling(window=short_window).mean()
    df['long_mavg'] = df['Close'].rolling(window=long_window).mean()

    df['signal'] = 0
    df.loc[df['short_mavg'] > df['long_mavg'], 'signal'] = 1
    df.loc[df['short_mavg'] < df['long_mavg'], 'signal'] = -1

    df['position'] = df['signal'].shift(1)
    df['trade'] = df['position'].diff().abs()

    df['daily_returns'] = df['Close'].pct_change()

    # Calculate trading costs and taxes
    df['cost'] = df['trade'] * df['Close'] * trading_cost
    df['tax'] = np.where(df['position'].diff() < 0, (df['daily_returns'] * df['Close']) * tax_rate, 0)
    df['net_daily_returns'] = df['daily_returns'] * df['position'] - df['cost'] - df['tax']

    df['cumulative_strategy_returns'] = (1 + df['net_daily_returns']).cumprod()

    return df['cumulative_strategy_returns'].iloc[-1]


def calculate_average_cumulative_return(stock_list, short_window, long_window, start_date, end_date, trading_cost, tax_rate):
    cumulative_returns = []

    for symbol in stock_list:
        if f'{symbol}.csv' not in os.listdir('data'):
            df = get_stock_data(symbol, start_date, end_date)
        else:
            df = pd.read_csv(f'data/{symbol}.csv')
        returns = calculate_strategy_returns(df.copy(), short_window, long_window, trading_cost, tax_rate)
        cumulative_returns.append(returns)

    return np.mean(cumulative_returns)


# Define stock list, start and end dates
stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'V', 'XOM', 'UNH', 'AMD', 'INTC', 'NFLX']
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2023, 3, 31)

# Define the search space for short and long windows
short_windows = range(5, 50)
long_windows = range(20, 100)

# Find the best lookback periods
best_return = -np.inf
best_short_window = None
best_long_window = None

for short_window in short_windows:
    for long_window in long_windows:
        if short_window >= long_window:
            continue

        avg_cumulative_return = calculate_average_cumulative_return(stock_list, short_window, long_window, start_date,
                                                                    end_date, trading_cost, tax_rate)

        if avg_cumulative_return > best_return:
            best_return = avg_cumulative_return
            best_short_window = short_window
            best_long_window = long_window

print(f"Best short window: {best_short_window}")
print(f"Best long window: {best_long_window}")
print(f"Best average cumulative return: {best_return}")
