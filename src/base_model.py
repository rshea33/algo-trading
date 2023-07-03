from datetime import datetime
from typing import *
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BaseModel:
    """Base class for trading models"""
    def __init__(
            self,
            ticker: str,
            start: Union[str, datetime],
            end: datetime,
            n_lags: Optional[int] = 1,
            laggables: Optional[List[Literal["Open", "High", "Low", "Close", "Adj Close", "Volume"]]] = None,
            
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.n_lags = n_lags
        self.laggables = laggables

        self._load_data()
    
    def backtest(
            self,
            initial_capital: float = 10000,
            plot: bool = True,
        ) -> None:
        """Backtest the strategy on the data"""
        try:
            self._strategy()
            self.df.dropna(inplace=True)
        except Exception as e:
            logging.error(f"Failed to run strategy for {self.ticker} from {self.start} to {self.end}")
            logging.error(e)
            raise e
        # add position that fills 0s with previous value
        self.df["position"] = self.df["signal"].ffill()
        self.df["portfolio_value"] = initial_capital + self.df["position"] * (1 + self.df["log_ret_1"])
        self.df["portfolio_value"] = self.df["portfolio_value"].cumsum()
        self.df["returns"] = self.df["portfolio_value"].pct_change()
        self.df["drawdown"] = self.df["portfolio_value"] - self.df["portfolio_value"].cummax()
        self.df["cummin"] = self.df["portfolio_value"].cummin()
        self.df["cummax"] = self.df["portfolio_value"].cummax()
        
        max_drawdown = self.df["drawdown"].min()
        sharpe_ratio = self.df["returns"].mean() / self.df["returns"].std()
        sortino_ratio = self.df["returns"].mean() / self.df["returns"][self.df["returns"] < 0].std()
        calmar_ratio = self.df["returns"].mean() / abs(max_drawdown)
        avg_return = self.df["returns"].mean()

        metrics = pd.DataFrame({
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "avg_return": avg_return,
            }, index=[0]
        )
        self.metrics = metrics
    
    def get_long_short_dates(self):
        """Returns a list of dates when the strategy is long and short"""
        long_dates = self.df[self.df["signal"] == 1].index
        short_dates = self.df[self.df["signal"] == -1].index
        return long_dates, short_dates
    
    def plot(
        self,
        save: Optional[str] = None,
) -> None:
        plt.figure(figsize=(12, 8))
        plt.plot(self.df["portfolio_value"], label="Portfolio Value")
        plt.plot(self.df["cummin"], label="Cumulative Minimum")
        plt.plot(self.df["cummax"], label="Cumulative Maximum")
        plt.legend()
        plt.tight_layout()
        if save:
            logging.info(f"Saving plot to visualizations/{save}.png")
            plt.savefig(f"visualizations/{save}.png")
        plt.show()
    
    def _strategy(self) -> None:
        """adds `signal` column to `self.df`"""
        raise NotImplementedError

    def _load_data(self) -> None:
        """Load data from Yahoo Finance, calculate log returns and add them to the dataframe"""
        logging.info(f"Loading data for {self.ticker} from {self.start} to {self.end}")
        try:
            self.df = yf.download(self.ticker, start=self.start, end=self.end)
        except Exception as e:
            logging.error(f"Failed to load data for {self.ticker} from {self.start} to {self.end}")
            raise e
        for n in range(1, self.n_lags + 1):
            self.df[f"log_ret_{n}"] = np.log(self.df["Adj Close"]) - np.log(self.df["Adj Close"].shift(n))
            if self.laggables:
                for lag in self.laggables:
                    self._lag_data(lag)
        self.df.dropna(inplace=True)

    def _lag_data(
        self,
        data: Literal["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    ) -> None:
        """Add lagged data to the dataframe"""
        for n in range(1, self.n_lags + 1):
            self.df[f"{data}_{n}"] = self.df[data].shift(n)
    