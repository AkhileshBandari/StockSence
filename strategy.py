import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

class SVMStrategy:

    def __init__(self, df):
        self.df = df.copy()
        self.model = None

    def generate_signals(self):

        # Ensure flat columns (fix yfinance MultiIndex)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)

        self.df["Return"] = self.df["Close"].pct_change()
        self.df["Target"] = np.where(self.df["Return"].shift(-1) > 0, 1, 0)
        self.df = self.df.dropna()

        X = self.df[["Open", "High", "Low", "Close", "Volume"]]
        y = self.df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = svm.SVC(kernel="rbf")
        self.model.fit(X_train, y_train)

        self.df["Signal"] = self.model.predict(X)

        return self.df