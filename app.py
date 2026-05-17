
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Nassau Candy — Profitability & Margin Dashboard",
    layout="wide",
)

BROWN = "#7B3F00"
PINK = "#FF69B4"
GRAY = "#808080"
GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"

PALETTE = {
    "Chocolate": BROWN,
    "Sugar": PINK,
    "Other": GRAY,
    "High": GREEN,
    "Medium": AMBER,
    "Low": RED,
}

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --primary-bg: #0f172a;
            --secondary-bg: #111827;
            --card-bg: rgba(255,255,255,0.08);
            --sidebar-bg: #111827;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --accent: #38bdf8;
            --border-color: rgba(255,255,255,0.12);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1e293b 100%);
            color: var(--text-primary);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 96%;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border-right: 1px solid var(--border-color);
        }

        [data-testid="stSidebar"] * {
            color: white !important;
        }

        h1, h2, h3, h4, h5 {
            color: white !important;
            font-weight: 700 !important;
        }

        p, label, div, span {
            color: #cbd5e1;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }

        [data-testid="stMetricLabel"] {
            color: #cbd5e1 !important;
            font-size: 14px !important;
        }

        [data-testid="stMetricValue"] {
            color: white !important;
            font-size: 30px !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(255,255,255,0.04);
            padding: 10px;
            border-radius: 18px;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #cbd5e1;
            border-radius: 12px;
            padding: 12px 18px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #38bdf8, #0ea5e9);
            color: white !important;
        }

        .stButton button {
            background: linear-gradient(135deg, #38bdf8, #0ea5e9);
            color: white !important;
            border: none;
            border-radius: 14px;
            padding: 0.7rem 1.3rem;
            font-weight: 600;
            transition: 0.3s ease;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(14,165,233,0.4);
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox div,
        .stMultiSelect div,
        textarea {
            background-color: rgba(255,255,255,0.06) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 12px !important;
        }

        .stSlider label,
        .stDateInput label,
        .stSelectbox label,
        .stTextInput label,
        .stMultiSelect label {
            color: white !important;
            font-weight: 600;
        }

        .stDataFrame {
            background: rgba(255,255,255,0.04);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            overflow: hidden;
        }

        .risk-box {
            border-left: 6px solid #ef4444;
            background: rgba(239,68,68,0.12);
            padding: 18px;
            border-radius: 16px;
            color: white !important;
            margin-top: 10px;
            border: 1px solid rgba(239,68,68,0.25);
        }

        .risk-box b {
            color: white !important;
        }

        .footer {
            color: #94a3b8;
            font-size: 0.9rem;
            text-align: center;
            padding-top: 30px;
            padding-bottom: 10px;
        }

        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.04);
            border-radius: 22px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 28px rgba(0,0,0,0.22);
        }

        .stAlert {
            border-radius: 16px;
        }

        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #0f172a;
        }

        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str = "data/processed/nassau_cleaned.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Quarter_Num"] = df["Quarter"].str.replace("Q", "", regex=False).astype(int)
    df["Year_Quarter"] = (
        df["Year"].astype(int).astype(str)
        + "-Q"
        + df["Quarter_Num"].astype(str)
    )
    return df


def weighted_margin(profit: pd.Series, sales: pd.Series) -> float:
    total_sales = sales.sum()
    return 0.0 if total_sales == 0 else (profit.sum() / total_sales) * 100


def percent(value: float) -> str:
    return f"{value:,.2f}%"


def currency(value: float) -> str:
    return f"${value:,.0f}"


def apply_plot_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Inter"),
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=35),
        title_font=dict(size=20, color="#ffffff"),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=13,
            font_color="#ffffff",
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        color="#cbd5e1",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        color="#cbd5e1",
    )

    return fig


df = load_data()

st.title("🍬 Nassau Candy — Profitability & Margin Dashboard")

st.sidebar.header("Filters")

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

selected_range = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

selected_divisions = st.sidebar.multiselect(
    "Division",
    options=["Chocolate", "Sugar", "Other"],
    default=["Chocolate", "Sugar", "Other"],
)

mask = (
    (df["Order Date"].dt.date >= start_date)
    & (df["Order Date"].dt.date <= end_date)
    & (df["Division"].isin(selected_divisions))
)

data = df.loc[mask].copy()

if data.empty:
    st.warning("No records found for selected filters.")
    st.stop()

total_revenue = data["Sales"].sum()
total_profit = data["Gross Profit"].sum()
company_margin = weighted_margin(data["Gross Profit"], data["Sales"])
product_count = data["Product Name"].nunique()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Revenue", currency(total_revenue))
k2.metric("Total Profit", currency(total_profit))
k3.metric("Gross Margin", percent(company_margin))
k4.metric("Products", f"{product_count:,}")

tab1, tab2 = st.tabs(
    [
        "Product Performance",
        "Division Insights",
    ]
)

with tab1:

    top_products = (
        data.groupby("Product Name", as_index=False)["Gross Profit"]
        .sum()
        .sort_values("Gross Profit", ascending=False)
        .head(15)
    )

    fig = px.bar(
        top_products,
        x="Gross Profit",
        y="Product Name",
        orientation="h",
        text="Gross Profit",
        color="Gross Profit",
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
    )

    st.plotly_chart(
        apply_plot_style(fig),
        use_container_width=True,
    )

    scatter = px.scatter(
        data,
        x="Sales",
        y="Gross_Margin_Pct",
        color="Division",
        size="Units",
        hover_name="Product Name",
        color_discrete_map=PALETTE,
        title="Sales vs Gross Margin %",
    )

    st.plotly_chart(
        apply_plot_style(scatter),
        use_container_width=True,
    )

    st.subheader("Product Data")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

with tab2:

    division_summary = (
        data.groupby("Division", as_index=False)
        .agg(
            Revenue=("Sales", "sum"),
            Profit=("Gross Profit", "sum"),
        )
    )

    div_fig = px.bar(
        division_summary,
        x="Division",
        y=["Revenue", "Profit"],
        barmode="group",
        color_discrete_sequence=[BROWN, PINK],
        title="Revenue vs Profit by Division",
    )

    st.plotly_chart(
        apply_plot_style(div_fig),
        use_container_width=True,
    )

    pie = px.pie(
        division_summary,
        names="Division",
        values="Profit",
        hole=0.45,
        color="Division",
        color_discrete_map=PALETTE,
        title="Profit Share by Division",
    )

    st.plotly_chart(
        apply_plot_style(pie),
        use_container_width=True,
    )

st.markdown(
    """
    <div class='footer'>
        Nassau Candy Distributor | Internal Analytics | Confidential
    </div>
    """,
    unsafe_allow_html=True,
)
