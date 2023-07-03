from typing import *
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from base_model import BaseModel


class ExampleStrategy(BaseModel):
    """Example strategy using standard RSI"""
    def __init__(self):
        super().__init__(
            ticker="INTC",
            start="2010-01-01",
            end="2020-01-01",
            n_lags=1,
        )

    def _strategy(self):
        """Calculate the strategy"""
        self.df["rsi"] = ta.rsi(self.df["Adj Close"], fillna=50)
        def _rsi_signal(rsi):
            if rsi > 70:
                return -1
            elif rsi < 30:
                return 1
            else:
                return np.nan
        signal = self.df["rsi"].apply(_rsi_signal)
        signal = signal.ffill()
        self.df["signal"] = signal

    
        
    
        
    
    

if __name__ == "__main__":
    strat = ExampleStrategy()

    strat.backtest()
    print(strat.df)
    print(strat.metrics)
    strat.plot()
    long, short = strat.get_long_short_dates()

    print(len(long))
    print(len(short))

    print(strat.df.describe())
