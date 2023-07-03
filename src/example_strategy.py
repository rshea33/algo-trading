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
            ticker="AAPL",
            start="2010-01-01",
            end="2020-01-01",
            n_lags=1,
        )

    def _strategy(self):
        """Calculate the strategy"""
        self.df["rsi"] = ta.rsi(self.df["Adj Close"], fillna=True)
        self.df["signal"] = np.where(self.df["rsi"] > 70, -1, 0)
        self.df["signal"] = np.where(self.df["rsi"] < 30, 1, self.df["signal"])
        self.df["signal"] = self.df["signal"].ffill().fillna(0)
    
        
    
    

if __name__ == "__main__":
    strat = ExampleStrategy()
    print(strat.df)
    metrics = strat.backtest()
    print(metrics)
