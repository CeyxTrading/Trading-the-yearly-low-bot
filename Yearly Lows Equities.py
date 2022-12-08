import os
import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.lib import Strategy
import yfinance as yf
pd.options.mode.chained_assignment = None
from ta.volatility import BollingerBands


def calculate_trend(values):
    if len(values) == 0:
        return 0, 0

    x = np.arange(1, len(values) + 1, 1)
    y = np.array(values)

    #  Handle nan values
    x_new = x[~np.isnan(y)]
    y_new = y[~np.isnan(y)]

    m, c = np.polyfit(x_new, y_new, 1)
    return m, c


def HIGH_BOLLINGER(df, window, std_dev):
    #  Bollinger Bands
    indicator_bb = BollingerBands(close=df['Close'], window=window, window_dev=std_dev)

    # Add Bollinger Bands features
    return indicator_bb.bollinger_hband()


class FiftyTwoWeekStrategy(Strategy):
    long_hold = 0
    low_price_threshold = 15 # percent
    trend_lb_period = 3
    trend_slope_threshold = 3
    bb_window = 20
    bb_std_dev = 2.4
    stop_loss = 15 # percent

    i = 0

    def init(self):
        super().init()

        #  Calculate Bollinger Bands
        self.bb_hband = self.I(HIGH_BOLLINGER, self.data.df, self.bb_window, self.bb_std_dev)


    def next(self):
        super().init()

        self.i += 1
        long_entry_signal = 0
        long_exit_signal = 0

        #  Skip the first year
        if self.i < 259:
            return

        #  Get the lookback price data
        close_lb = self.data.Close[self.i - 259:]

        #  Get the 52-week low
        fifty_two_week_low_price = np.min(np.array(close_lb))
        fifty_two_week_low_threshold = fifty_two_week_low_price * (1 + self.low_price_threshold/100)

        fifty_two_week_low_close_lb = self.data.Close[- self.trend_lb_period:]
        for check_price in fifty_two_week_low_close_lb:
            if check_price <= fifty_two_week_low_threshold:
                long_entry_signal += 1
                break

        #  Check if the price is trending up
        trend_close_lb = self.data.Close[- self.trend_lb_period:]

        #  Calculate trend slope
        m, c = calculate_trend(trend_close_lb)
        if m >= self.trend_slope_threshold:
            long_entry_signal += 1

        #  Long exit
        if self.data.High[-1] >= self.bb_hband[-1]:
            long_exit_signal += 1

        #  Perform trades
        if self.long_hold == 0 and long_entry_signal >= 2:
            self.buy(sl=self.data.Close[-1] * (1 - self.stop_loss/100))
            self.long_hold = 1
        elif self.long_hold == 1 and long_exit_signal >= 1:
            self.position.close()
            self.long_hold = 0


def run_backtest(df):
    bt = Backtest(df, FiftyTwoWeekStrategy, cash=10000, commission=.02, trade_on_close=True, exclusive_orders=True, hedging=False)
    stats = bt.run()
    bt.plot()
    return stats, bt


def load_data(symbol, period, interval):
    #  Download data
    # intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    #  periods:  1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    data = yf.download(tickers=symbol, period=period, interval=interval)
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    return df


# MAIN
if __name__ == '__main__':
    symbols = ['TSLA', 'COST', 'ORLY', 'PANW', 'MELI']

    total_returns = 0
    for symbol in symbols:
        print(f"Processing {symbol}")
        #  Download daily data
        df = load_data(symbol, period="2y", interval="1d")
        if df is None or df.empty:
            print(f"Dataframe for {symbol} is empty")
            continue

        #  run backtests
        stats = run_backtest(df)

        #  Store total return
        total_returns += stats[0]['Return [%]']
        print(f"Return for {symbol}: {stats[0]['Return [%]']}")

    #  Print average return for all symbols
    total_avg_return = total_returns / len(symbols)
    print(f'Average return for all symbols: {"{:.2f}".format(total_avg_return)}')

