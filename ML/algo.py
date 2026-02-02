import io
import requests
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

# -----------------------------
# Load data
# -----------------------------
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d51iMGfp_t0QpO30Lym-dw/automobile-sales.csv"
df = pd.read_csv(io.StringIO(requests.get(URL).text))

# helper: safely pick a column name that exists (for datasets with slightly different headers)
def pick_col(options):
    for c in options:
        if c in df.columns:
            return c
    return None

# columns (robust)
YEAR_COL = pick_col(["Year", "year"])
MONTH_COL = pick_col(["Month", "month"])
SALES_COL = pick_col(["Automobile_Sales", "Automobile Sales", "automobile_sales"])
AD_COL = pick_col(["Advertising_Expenditure", "Advertising Expenditure", "advertising_expenditure"])
VEH_COL = pick_col(["Vehicle_Type", "Vehicle Type", "vehicle_type"])
REC_COL = pick_col(["Recession", "recession"])
UNEMP_COL = pick_col(["Unemployment_Rate", "Unemployment rate", "unemployment_rate"])
GDP_COL = pick_col(["GDP", "gdp"])

# -----------------------------
# Dash app
# -----------------------------
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[
    html.H1(
        "Automobile Sales Dashboard",
        style={"textAlign": "center", "color": "#503D36", "fontSize": 26}
    ),

    # Dropdowns row
    html.Div([
        html.Div([
            html.H2("Select Report Type:", style={"marginRight": "2em"}),
            dcc.Dropdown(
                id="dropdown-statistics",
                options=[
                    {"label": "Yearly Statistics", "value": "Yearly Statistics"},
                    {"label": "Recession Period Statistics", "value": "Recession Period Statistics"},
                ],
                value="Yearly Statistics",
                clearable=False
            )
        ], style={"width": "48%"}),

        html.Div([
            html.H2("Select Year:", style={"marginRight": "2em"}),
            dcc.Dropdown(
                id="select-year",
                options=[{"label": int(y), "value": int(y)} for y in sorted(df[YEAR_COL].unique())],
                value=int(sorted(df[YEAR_COL].unique())[0]),
                clearable=False,
                disabled=False
            )
        ], style={"width": "48%"}),
    ], style={"display": "flex", "justifyContent": "space-between"}),

    html.Br(),

    # 4 plots in 2 rows x 2 columns
    html.Div([
        html.Div([dcc.Graph(id="plot1")], style={"width": "50%"}),
        html.Div([dcc.Graph(id="plot2")], style={"width": "50%"}),
    ], style={"display": "flex"}),

    html.Div([
        html.Div([dcc.Graph(id="plot3")], style={"width": "50%"}),
        html.Div([dcc.Graph(id="plot4")], style={"width": "50%"}),
    ], style={"display": "flex"}),
])


# -----------------------------
# Callback 1: enable/disable year dropdown
# -----------------------------
@app.callback(
    Output("select-year", "disabled"),
    Input("dropdown-statistics", "value")
)
def update_year_dropdown(selected_statistics):
    # enabled when Yearly Statistics -> disabled should be False
    # disabled when Recession Period Statistics -> disabled should be True
    if selected_statistics == "Yearly Statistics":
        return False
    return True


# -----------------------------
# Callback 2: update 4 plots
# -----------------------------
@app.callback(
    [Output("plot1", "figure"),
     Output("plot2", "figure"),
     Output("plot3", "figure"),
     Output("plot4", "figure")],
    [Input("dropdown-statistics", "value"),
     Input("select-year", "value")]
)
def update_plots(selected_statistics, selected_year):

    if selected_statistics == "Yearly Statistics":
        dff = df[df[YEAR_COL] == selected_year]

        # Plot 1: Monthly average sales (if Month exists), else yearly sales trend
        if MONTH_COL:
            m1 = dff.groupby(MONTH_COL, as_index=False)[SALES_COL].mean()
            fig1 = px.line(m1, x=MONTH_COL, y=SALES_COL,
                           title=f"Average Automobile Sales by Month ({selected_year})",
                           markers=True)
        else:
            y1 = df.groupby(YEAR_COL, as_index=False)[SALES_COL].mean()
            fig1 = px.line(y1, x=YEAR_COL, y=SALES_COL,
                           title="Average Automobile Sales by Year",
                           markers=True)

        # Plot 2: Avg sales by vehicle type (year)
        if VEH_COL:
            v2 = dff.groupby(VEH_COL, as_index=False)[SALES_COL].mean()
            fig2 = px.bar(v2, x=VEH_COL, y=SALES_COL,
                          title=f"Average Sales by Vehicle Type ({selected_year})")
        else:
            fig2 = px.bar(title="Vehicle type column not found")

        # Plot 3: Ad spend by vehicle type (year) — pie
        if VEH_COL and AD_COL:
            v3 = dff.groupby(VEH_COL, as_index=False)[AD_COL].sum()
            fig3 = px.pie(v3, names=VEH_COL, values=AD_COL,
                          title=f"Advertising Expenditure Share by Vehicle Type ({selected_year})")
        else:
            fig3 = px.pie(title="Advertising/Vehicle type column not found")

        # Plot 4: Ad spend vs sales (year) — scatter
        if AD_COL:
            fig4 = px.scatter(dff, x=AD_COL, y=SALES_COL,
                              title=f"Advertising vs Sales ({selected_year})",
                              trendline="ols" if len(dff) > 2 else None)
        else:
            fig4 = px.scatter(title="Advertising column not found")

        return fig1, fig2, fig3, fig4

    # ---------------- Recession Period Statistics ----------------
    rec_df = df[df[REC_COL] == 1] if REC_COL else df

    # Plot 1: Avg sales by year during recession
    y1 = rec_df.groupby(YEAR_COL, as_index=False)[SALES_COL].mean()
    fig1 = px.line(y1, x=YEAR_COL, y=SALES_COL,
                   title="Average Automobile Sales during Recession (Year-wise)",
                   markers=True)

    # Plot 2: Avg sales by vehicle type during recession
    if VEH_COL:
        v2 = rec_df.groupby(VEH_COL, as_index=False)[SALES_COL].mean()
        fig2 = px.bar(v2, x=VEH_COL, y=SALES_COL,
                      title="Average Sales by Vehicle Type during Recession")
    else:
        fig2 = px.bar(title="Vehicle type column not found")

    # Plot 3: Total ad expenditure share (Recession vs Non-Recession)
    if AD_COL and REC_COL:
        totals = df.groupby(REC_COL, as_index=False)[AD_COL].sum()
        totals[REC_COL] = totals[REC_COL].map({0: "Non-Recession", 1: "Recession"})
        fig3 = px.pie(totals, names=REC_COL, values=AD_COL,
                      title="Advertising Expenditure: Recession vs Non-Recession")
    else:
        fig3 = px.pie(title="Advertising/Recession column not found")

    # Plot 4: Unemployment vs sales during recession (if unemployment exists); else GDP vs sales
    x_col = UNEMP_COL if UNEMP_COL else GDP_COL
    if x_col:
        fig4 = px.scatter(rec_df, x=x_col, y=SALES_COL,
                          title=f"{x_col} vs Automobile Sales during Recession",
                          trendline="ols" if len(rec_df) > 2 else None)
    else:
        fig4 = px.scatter(title="No unemployment/GDP column found")

    return fig1, fig2, fig3, fig4


if __name__ == "__main__":
    app.run(debug=True)
