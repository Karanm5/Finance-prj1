# dashfolder/app.py
import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

# ===============================
# CONFIGURATION
# ===============================
RESULTS_FOLDER = "evaluation/results"
METRICS_FILE = os.path.join(RESULTS_FOLDER, "evaluation_metrics.csv")
DATA_FILE = "data/spy_features_regime.csv"

# ===============================
# LOAD DATA
# ===============================
metrics_df = pd.read_csv(METRICS_FILE)
df = pd.read_csv(DATA_FILE)

# Load net worth, cash, shares from saved CSV (or recalc from df if needed)
# Here we assume portfolio_value.csv exists, else use eval code to generate arrays
portfolio_value_file = os.path.join(RESULTS_FOLDER, "portfolio_value.csv")
if os.path.exists(portfolio_value_file):
    portfolio_df = pd.read_csv(portfolio_value_file)
    net_worths = portfolio_df["Net_Worth"].values
    cash_history = portfolio_df["Cash"].values
    shares_value = portfolio_df["Shares_Value"].values
else:
    # Recompute from df if not saved
    net_worths = df["Close"].values  # placeholder
    cash_history = net_worths * 0
    shares_value = net_worths * 0

steps = list(range(len(net_worths)))

# ===============================
# DASH APP
# ===============================
app = Dash(__name__)
app.title = "Trading Agent Dashboard"

app.layout = html.Div(
    style={"margin": "50px"},
    children=[
        html.H1("RL Trading Agent Performance", style={"textAlign": "center"}),

        html.H2("Summary Metrics"),
        html.Table([
            html.Tr([html.Th(k), html.Td(f"{v:.2f}")]) for k, v in metrics_df.iloc[0].items()
        ], style={"width": "50%", "margin": "auto", "fontSize": "18px"}),

        html.H2("Portfolio Value Over Time"),
        dcc.Graph(
            figure={
                "data": [
                    go.Scatter(x=steps, y=net_worths, mode="lines", name="Net Worth"),
                    go.Scatter(x=steps, y=cash_history, mode="lines", name="Cash"),
                    go.Scatter(x=steps, y=shares_value, mode="lines", name="Shares Value")
                ],
                "layout": go.Layout(
                    xaxis_title="Step",
                    yaxis_title="Value ($)",
                    template="plotly_dark"
                )
            }
        ),

        html.H2("Drawdown Over Time"),
        dcc.Graph(
            figure={
                "data": [
                    go.Scatter(
                        x=steps,
                        y=[(max(net_worths[:i+1]) - net_worths[i]) / max(net_worths[:i+1])
                            for i in range(len(net_worths))],
                        mode="lines", 
                        name="Drawdown", 
                        line=dict(color="red"))
                ],
                "layout": go.Layout(
                    xaxis_title="Step",
                    yaxis_title="Drawdown",
                    template="plotly_dark"
                )
            }
        ),
    ]
)

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
