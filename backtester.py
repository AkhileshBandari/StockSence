import numpy as np

class Backtester:

    def __init__(self, df,
                 initial_capital=100000,
                 risk_per_trade=0.02,
                 transaction_cost=0.001):

        self.df = df.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.transaction_cost = transaction_cost
        self.position = 0
        self.entry_price = 0
        self.equity_curve = []
        self.trade_log = []

    def run(self):

        for i in range(len(self.df) - 1):

            price = self.df["Close"].iloc[i]
            signal = self.df["Signal"].iloc[i]

            # Ensure numeric scalar
            if hasattr(price, "values"):
                price = price.values[0]

            price = float(price)

            if price <= 0:
                continue

            position_size = int((self.capital * self.risk_per_trade) / price)

            # BUY
            if signal == 1 and self.position == 0 and position_size > 0:
                self.position = position_size
                self.entry_price = price
                self.trade_log.append(("BUY", self.df.index[i], price))

            # SELL
            elif signal == 0 and self.position > 0:
                profit = (price - self.entry_price) * self.position
                cost = price * self.position * self.transaction_cost
                self.capital += profit - cost

                self.trade_log.append(
                    ("SELL", self.df.index[i], price, profit - cost)
                )

                self.position = 0

            self.equity_curve.append(self.capital)

        return self._performance_metrics()

    def _performance_metrics(self):

        equity = np.array(self.equity_curve)

        if len(equity) < 2:
            return {
                "final_capital": self.capital,
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "equity_curve": equity,
                "trade_log": self.trade_log
            }

        returns = np.diff(equity) / equity[:-1]

        sharpe = (
            np.mean(returns) / np.std(returns)
            if np.std(returns) != 0 else 0
        )

        drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
        max_drawdown = drawdown.min()

        return {
            "final_capital": self.capital,
            "total_return": (self.capital / self.initial_capital - 1) * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "equity_curve": equity,
            "trade_log": self.trade_log
        }