import pandas as pd
import yfinance as yf
import datetime

# Define stock, start and end dates
symbol = 'AAPL'
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2023, 3, 31)

# Retrieve stock data
df = yf.download(symbol, start=start_date, end=end_date)

# Calculate moving averages
short_window = 20
long_window = 50
df['short_mavg'] = df['Close'].rolling(window=short_window).mean()
df['long_mavg'] = df['Close'].rolling(window=long_window).mean()

# Identify buy and sell signals
df['signal'] = 0
df.loc[df['short_mavg'] > df['long_mavg'], 'signal'] = 1
df.loc[df['short_mavg'] < df['long_mavg'], 'signal'] = -1

# Calculate daily returns and strategy returns
df['daily_returns'] = df['Close'].pct_change()
df['strategy_returns'] = df['daily_returns'] * df['signal'].shift(1)

# Calculate cumulative returns
df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()

print(df.tail())
