import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dash
from dash import dcc, html
import plotly.graph_objs as go

# =========================
# CONFIGURATION
# =========================
SYMBOL = "RELIANCE.NS"
START_DATE = "2021-01-01"
END_DATE = "2023-01-01"
INITIAL_CAPITAL = 100000
POSITION_SIZE = 1

# =========================
# DOWNLOAD DATA
# =========================
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

df["Return"] = df["Close"].pct_change()
df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
df = df.dropna()

# =========================
# MODEL TRAINING
# =========================
X = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = svm.SVC(kernel="rbf")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# =========================
# PAPER TRADING ENGINE
# =========================
capital = INITIAL_CAPITAL
position = 0
entry_price = 0
equity_curve = []
trade_log = []

test_df = df.iloc[-len(X_test):].copy()
test_df["Prediction"] = predictions

for i in range(len(test_df) - 1):

    price = test_df["Close"].iloc[i]
    signal = test_df["Prediction"].iloc[i]

    # BUY SIGNAL
    if signal == 1 and position == 0:
        position = POSITION_SIZE
        entry_price = price
        trade_log.append(("BUY", test_df.index[i], price))

    # SELL SIGNAL
    elif signal == 0 and position > 0:
        profit = (price - entry_price) * position
        capital += profit
        trade_log.append(("SELL", test_df.index[i], price, profit))
        position = 0

    equity_curve.append(capital)

# Close open position
if position > 0:
    final_price = test_df["Close"].iloc[-1]
    profit = (final_price - entry_price) * position
    capital += profit
    trade_log.append(("FINAL SELL", test_df.index[-1], final_price, profit))
    position = 0

final_profit = capital - INITIAL_CAPITAL

# =========================
# DASHBOARD
# =========================
app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1("Stock Market SVM + Paper Trading System",
            style={"textAlign": "center"}),

    html.H3(f"Stock: {SYMBOL}"),
    html.H3(f"Model Accuracy: {round(accuracy*100,2)}%"),
    html.H3(f"Initial Capital: ₹{INITIAL_CAPITAL}"),
    html.H3(f"Final Capital: ₹{round(capital,2)}"),
    html.H3(f"Total P&L: ₹{round(final_profit,2)}"),

    dcc.Graph(
        figure={
            "data": [
                go.Scatter(
                    x=test_df.index[:len(equity_curve)],
                    y=equity_curve,
                    mode="lines",
                    name="Equity Curve"
                )
            ],
            "layout": go.Layout(
                title="Equity Curve (Paper Trading)",
                xaxis={"title": "Date"},
                yaxis={"title": "Capital"}
            )
        }
    ),

    dcc.Graph(
        figure={
            "data": [
                go.Scatter(
                    x=test_df.index,
                    y=test_df["Close"],
                    mode="lines",
                    name="Stock Price"
                )
            ],
            "layout": go.Layout(
                title="Stock Closing Price",
                xaxis={"title": "Date"},
                yaxis={"title": "Price"}
            )
        }
    )
])

if __name__ == "__main__":
    app.run(debug=True)