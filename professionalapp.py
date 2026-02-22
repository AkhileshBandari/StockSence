import pandas as pd
import numpy as np
import yfinance as yf
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from sklearn import svm
from sklearn.model_selection import train_test_split


# 100+ NSE STOCK LIST

STOCK_LIST = [
"RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS",
"HINDUNILVR.NS","ITC.NS","LT.NS","KOTAKBANK.NS","AXISBANK.NS","ASIANPAINT.NS",
"MARUTI.NS","BAJFINANCE.NS","HCLTECH.NS","WIPRO.NS","ONGC.NS","TITAN.NS",
"ULTRACEMCO.NS","POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS","NTPC.NS",
"TATAMOTORS.NS","JSWSTEEL.NS","SUNPHARMA.NS","DRREDDY.NS","TECHM.NS",
"GRASIM.NS","COALINDIA.NS","BPCL.NS","INDUSINDBK.NS","BRITANNIA.NS",
"EICHERMOT.NS","DIVISLAB.NS","BAJAJFINSV.NS","CIPLA.NS","HEROMOTOCO.NS",
"SBILIFE.NS","APOLLOHOSP.NS","HDFCLIFE.NS","UPL.NS","TATASTEEL.NS",
"SHREECEM.NS","HINDALCO.NS","BAJAJ-AUTO.NS","NESTLEIND.NS","M&M.NS",
"PIDILITIND.NS","DABUR.NS","DMART.NS","ICICIPRULI.NS","GODREJCP.NS",
"SIEMENS.NS","AMBUJACEM.NS","ABB.NS","VEDL.NS","TATACONSUM.NS",
"SBICARD.NS","BANDHANBNK.NS","INDIGO.NS","NAUKRI.NS","COLPAL.NS",
"BIOCON.NS","HAVELLS.NS","PAGEIND.NS","MCDOWELL-N.NS","GAIL.NS",
"NMDC.NS","PEL.NS","ICICIGI.NS","TORNTPHARM.NS","LUPIN.NS",
"CANBK.NS","BANKBARODA.NS","PNB.NS","YESBANK.NS","IDEA.NS",
"SAIL.NS","ZOMATO.NS","PAYTM.NS","NYKAA.NS","POLYCAB.NS",
"CHOLAFIN.NS","TRENT.NS","AUBANK.NS","IDFCFIRSTB.NS","TVSMOTOR.NS",
"BEL.NS","IRCTC.NS","TATAPOWER.NS","DLF.NS","MFSL.NS",
"MARICO.NS","ADANIGREEN.NS","ADANIPOWER.NS","CROMPTON.NS","EXIDEIND.NS"
]


# DASH APP

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1("Multi-Stock Professional Backtester",
            style={"textAlign": "center"}),

    dcc.Dropdown(
        id="stock-dropdown",
        options=[{"label": stock, "value": stock} for stock in STOCK_LIST],
        value="RELIANCE.NS",
        searchable=True,
        style={"width": "60%", "margin": "auto"}
    ),

    html.Div(id="metrics"),

    dcc.Graph(id="price-chart"),

    dcc.Graph(id="equity-chart")
])


# BACKTEST FUNCTION

def run_backtest(symbol):

    df = yf.download(symbol, start="2021-01-01", end="2023-01-01")

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return"] = df["Close"].pct_change()
    df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
    df = df.dropna()

    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = svm.SVC(kernel="rbf")
    model.fit(X_train, y_train)

    df["Signal"] = model.predict(X)

    capital = 100000
    position = 0
    entry_price = 0
    equity_curve = []

    for i in range(len(df)-1):

        price = float(df["Close"].iloc[i])
        signal = df["Signal"].iloc[i]

        position_size = int((capital * 0.02) / price)

        if signal == 1 and position == 0 and position_size > 0:
            position = position_size
            entry_price = price

        elif signal == 0 and position > 0:
            profit = (price - entry_price) * position
            cost = price * position * 0.001
            capital += profit - cost
            position = 0

        equity_curve.append(capital)

    total_return = (capital / 100000 - 1) * 100

    return df, equity_curve, capital, total_return


# CALLBACK

@app.callback(
    Output("price-chart", "figure"),
    Output("equity-chart", "figure"),
    Output("metrics", "children"),
    Input("stock-dropdown", "value")
)
def update_dashboard(selected_stock):

    result = run_backtest(selected_stock)

    if result is None:
        return {}, {}, "No data available"

    df, equity_curve, final_capital, total_return = result

    price_fig = {
        "data": [
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Closing Price"
            )
        ],
        "layout": go.Layout(title=f"{selected_stock} Price")
    }

    equity_fig = {
        "data": [
            go.Scatter(
                y=equity_curve,
                mode="lines",
                name="Equity Curve"
            )
        ],
        "layout": go.Layout(title="Equity Curve")
    }

    metrics = html.Div([
        html.H3(f"Final Capital: ₹{round(final_capital,2)}"),
        html.H3(f"Total Return: {round(total_return,2)}%")
    ])

    return price_fig, equity_fig, metrics


if __name__ == "__main__":
    app.run(debug=True)