from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("data/processed/nassau_cleaned.csv")
OUTPUT_DIR = Path("eda_charts")

DIVISION_COLORS = {
    "Chocolate": "#7B3F00",
    "Sugar": "#FF69B4",
    "Other": "#808080",
}

MARGIN_COLORS = {
    "High": "#2E7D32",
    "Medium": "#F9A825",
    "Low": "#C62828",
}


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2f}%"


def weighted_margin(profit: pd.Series, sales: pd.Series) -> float:
    total_sales = sales.sum()
    if total_sales == 0:
        return 0.0
    return (profit.sum() / total_sales) * 100


def wrapped_label(label: str, width: int = 22) -> str:
    return "\n".join(wrap(str(label), width=width))


def print_chart_insight(chart_title: str, line_1: str, line_2: str) -> None:
    print(f"\n{chart_title}")
    print(f"Insight 1: {line_1}")
    print(f"Insight 2: {line_2}")


def save_chart(fig: plt.Figure, filename: str, insight_1: str, insight_2: str) -> None:
    fig.text(0.01, 0.035, f"Insight 1: {insight_1}", fontsize=10, ha="left")
    fig.text(0.01, 0.012, f"Insight 2: {insight_2}", fontsize=10, ha="left")
    fig.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def label_barh(ax: plt.Axes, fmt_func=money, padding: float = 0.01) -> None:
    x_max = ax.get_xlim()[1]
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + (x_max * padding),
            patch.get_y() + patch.get_height() / 2,
            fmt_func(width),
            va="center",
            ha="left",
            fontsize=9,
        )


def label_bars(ax: plt.Axes, fmt_func=pct) -> None:
    y_max = ax.get_ylim()[1]
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + y_max * 0.015,
            fmt_func(height),
            va="bottom",
            ha="center",
            fontsize=9,
        )


def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    product = (
        df.groupby(["Division", "Product Name"], as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Cost=("Cost", "sum"),
            Gross_Profit=("Gross Profit", "sum"),
            Units=("Units", "sum"),
        )
    )
    product["Gross_Margin_Pct"] = ((product["Gross_Profit"] / product["Sales"]) * 100).round(2)
    product["Profit_Per_Unit"] = (product["Gross_Profit"] / product["Units"]).round(2)
    product["Profit_Contribution_Pct"] = (
        (product["Gross_Profit"] / product["Gross_Profit"].sum()) * 100
    ).round(2)
    return product


def chart_1_revenue_by_division(df: pd.DataFrame) -> None:
    revenue = df.groupby("Division", as_index=False)["Sales"].sum().sort_values("Sales", ascending=True)
    top = revenue.iloc[-1]
    bottom = revenue.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(revenue["Division"], revenue["Sales"], color=[DIVISION_COLORS.get(x, "#333333") for x in revenue["Division"]])
    ax.set_title("Total Revenue by Division")
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("Division")
    ax.xaxis.set_major_formatter(lambda x, _: money(x))
    label_barh(ax, money)

    insight_1 = f"{top['Division']} leads revenue at {money(top['Sales'])}, showing the largest commercial footprint."
    insight_2 = f"{bottom['Division']} is the smallest revenue contributor at {money(bottom['Sales'])}."
    save_chart(fig, "01_total_revenue_by_division.png", insight_1, insight_2)
    print_chart_insight("Chart 1 - Total Revenue by Division", insight_1, insight_2)


def chart_2_margin_by_division(df: pd.DataFrame) -> None:
    margin = (
        df.groupby("Division")
        .apply(lambda x: weighted_margin(x["Gross Profit"], x["Sales"]))
        .reset_index(name="Gross_Margin_Pct")
        .sort_values("Gross_Margin_Pct", ascending=False)
    )
    company_avg = weighted_margin(df["Gross Profit"], df["Sales"])
    best = margin.iloc[0]
    worst = margin.iloc[-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=margin, x="Division", y="Gross_Margin_Pct", ax=ax, palette=DIVISION_COLORS, hue="Division", legend=False)
    ax.axhline(company_avg, color="red", linestyle="--", linewidth=1.5, label=f"Company Avg: {pct(company_avg)}")
    ax.set_title("Gross Margin % by Division")
    ax.set_xlabel("Division")
    ax.set_ylabel("Gross Margin %")
    ax.legend()
    label_bars(ax, pct)

    insight_1 = f"{best['Division']} has the highest weighted margin at {pct(best['Gross_Margin_Pct'])}."
    insight_2 = f"{worst['Division']} trails at {pct(worst['Gross_Margin_Pct'])}; company average is {pct(company_avg)}."
    save_chart(fig, "02_gross_margin_by_division.png", insight_1, insight_2)
    print_chart_insight("Chart 2 - Gross Margin % by Division", insight_1, insight_2)


def chart_3_revenue_vs_profit(product: pd.DataFrame) -> None:
    top5 = product.nlargest(5, "Gross_Profit")
    bottom5 = product.nsmallest(5, "Gross_Profit")
    annotated = pd.concat([top5, bottom5]).drop_duplicates(subset=["Product Name"])

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=product,
        x="Sales",
        y="Gross_Profit",
        hue="Division",
        palette=DIVISION_COLORS,
        s=120,
        ax=ax,
    )
    max_sales = product["Sales"].max() * 1.05
    ax.plot([0, max_sales], [0, max_sales * 0.30], color="black", linestyle="--", linewidth=1.2, label="30% Profit Reference")
    for _, row in annotated.iterrows():
        ax.annotate(
            wrapped_label(row["Product Name"], 16),
            (row["Sales"], row["Gross_Profit"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Revenue vs Gross Profit by Product")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Total Gross Profit")
    ax.xaxis.set_major_formatter(lambda x, _: money(x))
    ax.yaxis.set_major_formatter(lambda y, _: money(y))
    ax.legend()

    leader = product.loc[product["Gross_Profit"].idxmax()]
    laggard = product.loc[product["Gross_Profit"].idxmin()]
    insight_1 = f"{leader['Product Name']} delivers the highest gross profit at {money(leader['Gross_Profit'])}."
    insight_2 = f"{laggard['Product Name']} has the lowest gross profit at {money(laggard['Gross_Profit'])}."
    save_chart(fig, "03_revenue_vs_gross_profit_scatter.png", insight_1, insight_2)
    print_chart_insight("Chart 3 - Revenue vs Gross Profit Scatter", insight_1, insight_2)


def chart_4_margin_distribution(df: pd.DataFrame) -> None:
    outliers = []
    for division, group in df.groupby("Division"):
        q1 = group["Gross_Margin_Pct"].quantile(0.25)
        q3 = group["Gross_Margin_Pct"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rows = group[(group["Gross_Margin_Pct"] < lower) | (group["Gross_Margin_Pct"] > upper)]
        outliers.append(outlier_rows)
    outlier_df = pd.concat(outliers) if outliers else pd.DataFrame(columns=df.columns)
    labeled_outliers = (
        outlier_df.sort_values("Gross_Margin_Pct")
        .drop_duplicates(["Division", "Product Name"])
        .groupby("Division", as_index=False)
        .head(5)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Division", y="Gross_Margin_Pct", hue="Division", palette=DIVISION_COLORS, ax=ax, legend=False)
    sns.stripplot(data=labeled_outliers, x="Division", y="Gross_Margin_Pct", color="black", size=5, jitter=0.18, ax=ax)
    division_positions = {label.get_text(): idx for idx, label in enumerate(ax.get_xticklabels())}
    for _, row in labeled_outliers.iterrows():
        ax.annotate(
            wrapped_label(row["Product Name"], 14),
            (division_positions[row["Division"]], row["Gross_Margin_Pct"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    medians = df.groupby("Division")["Gross_Margin_Pct"].median()
    for division, median in medians.items():
        ax.text(division_positions[division], median, pct(median), ha="center", va="bottom", fontsize=9, color="white")
    ax.set_title("Gross Margin % Distribution by Division")
    ax.set_xlabel("Division")
    ax.set_ylabel("Gross Margin %")

    widest = df.groupby("Division")["Gross_Margin_Pct"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    insight_1 = f"{widest.index[0]} has the widest margin spread at {pct(widest.iloc[0])}."
    insight_2 = f"{len(labeled_outliers)} distinct product outlier labels are shown across divisions."
    save_chart(fig, "04_gross_margin_distribution.png", insight_1, insight_2)
    print_chart_insight("Chart 4 - Gross Margin % Distribution", insight_1, insight_2)


def chart_5_top_bottom_margin_products(product: pd.DataFrame) -> None:
    top10 = product.nlargest(10, "Gross_Margin_Pct").sort_values("Gross_Margin_Pct")
    bottom10 = product.nsmallest(10, "Gross_Margin_Pct").sort_values("Gross_Margin_Pct")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    axes[0].barh(top10["Product Name"].map(wrapped_label), top10["Gross_Margin_Pct"], color="#2E7D32")
    axes[0].set_title("Top 10 Products by Gross Margin %")
    axes[0].set_xlabel("Gross Margin %")
    axes[0].set_ylabel("Product Name")
    label_barh(axes[0], pct)

    axes[1].barh(bottom10["Product Name"].map(wrapped_label), bottom10["Gross_Margin_Pct"], color="#C62828")
    axes[1].set_title("Bottom 10 Products by Gross Margin %")
    axes[1].set_xlabel("Gross Margin %")
    axes[1].set_ylabel("Product Name")
    label_barh(axes[1], pct)

    fig.suptitle("Top 10 and Bottom 10 Products by Gross Margin %")
    best = product.loc[product["Gross_Margin_Pct"].idxmax()]
    worst = product.loc[product["Gross_Margin_Pct"].idxmin()]
    insight_1 = f"{best['Product Name']} has the strongest product margin at {pct(best['Gross_Margin_Pct'])}."
    insight_2 = f"{worst['Product Name']} has the weakest product margin at {pct(worst['Gross_Margin_Pct'])}."
    save_chart(fig, "05_top_bottom_products_by_margin.png", insight_1, insight_2)
    print_chart_insight("Chart 5 - Top and Bottom Products by Gross Margin %", insight_1, insight_2)


def chart_6_quarterly_revenue(df: pd.DataFrame) -> None:
    trend = df.copy()
    trend["Quarter_Num"] = trend["Quarter"].str.replace("Q", "", regex=False).astype(int)
    trend["Year_Quarter"] = trend["Year"].astype(int).astype(str) + "-Q" + trend["Quarter_Num"].astype(str)
    trend = (
        trend.groupby(["Year", "Quarter_Num", "Year_Quarter", "Division"], as_index=False)["Sales"]
        .sum()
        .sort_values(["Year", "Quarter_Num"])
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=trend, x="Year_Quarter", y="Sales", hue="Division", palette=DIVISION_COLORS, marker="o", ax=ax)
    for _, row in trend.iterrows():
        ax.text(row["Year_Quarter"], row["Sales"], money(row["Sales"]), fontsize=8, ha="center", va="bottom", rotation=35)
    ax.set_title("Quarterly Revenue Trend by Division")
    ax.set_xlabel("Year-Quarter")
    ax.set_ylabel("Total Sales")
    ax.yaxis.set_major_formatter(lambda y, _: money(y))
    ax.tick_params(axis="x", rotation=35)

    quarterly_total = trend.groupby("Year_Quarter")["Sales"].sum()
    insight_1 = f"Peak total quarterly revenue occurs in {quarterly_total.idxmax()} at {money(quarterly_total.max())}."
    insight_2 = f"Lowest total quarterly revenue occurs in {quarterly_total.idxmin()} at {money(quarterly_total.min())}."
    save_chart(fig, "06_quarterly_revenue_trend.png", insight_1, insight_2)
    print_chart_insight("Chart 6 - Quarterly Revenue Trend", insight_1, insight_2)


def chart_7_pareto_profit(product: pd.DataFrame) -> None:
    pareto = product.sort_values("Gross_Profit", ascending=False).reset_index(drop=True)
    pareto["Cumulative_Profit_Pct"] = (pareto["Gross_Profit"].cumsum() / pareto["Gross_Profit"].sum()) * 100
    threshold_row = pareto[pareto["Cumulative_Profit_Pct"] >= 80].iloc[0]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    labels = pareto["Product Name"].map(lambda x: wrapped_label(x, 12))
    bars = ax1.bar(labels, pareto["Gross_Profit"], color="#4E79A7")
    ax1.set_title("Pareto Chart: Product Profit Contribution")
    ax1.set_xlabel("Product Name")
    ax1.set_ylabel("Gross Profit")
    ax1.yaxis.set_major_formatter(lambda y, _: money(y))
    ax1.tick_params(axis="x", rotation=70)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), money(bar.get_height()), ha="center", va="bottom", fontsize=8, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(labels, pareto["Cumulative_Profit_Pct"], color="#E15759", marker="o", linewidth=2)
    ax2.axhline(80, color="black", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Cumulative Profit Contribution %")
    ax2.set_ylim(0, 110)
    for idx, value in enumerate(pareto["Cumulative_Profit_Pct"]):
        ax2.text(idx, value + 2, pct(value), ha="center", va="bottom", fontsize=7)

    insight_1 = f"The 80% profit threshold is reached by {threshold_row['Product Name']} in rank {int(threshold_row.name) + 1}."
    insight_2 = f"The top product contributes {pct(pareto.loc[0, 'Profit_Contribution_Pct'])} of gross profit."
    save_chart(fig, "07_pareto_profit_contribution.png", insight_1, insight_2)
    print_chart_insight("Chart 7 - Pareto Profit Contribution", insight_1, insight_2)


def chart_8_cost_vs_sales(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=df,
        x="Cost",
        y="Sales",
        hue="Margin_Flag",
        hue_order=["High", "Medium", "Low"],
        palette=MARGIN_COLORS,
        alpha=0.45,
        s=35,
        ax=ax,
    )
    for margin_flag, group in df.groupby("Margin_Flag"):
        if len(group) >= 2:
            sns.regplot(data=group, x="Cost", y="Sales", scatter=False, ax=ax, color=MARGIN_COLORS.get(margin_flag, "#333333"), ci=None)
        ax.text(
            group["Cost"].median(),
            group["Sales"].median(),
            f"{margin_flag}: {len(group):,}",
            fontsize=10,
            weight="bold",
            color=MARGIN_COLORS.get(margin_flag, "#333333"),
        )
    ax.set_title("Cost vs Sales with Margin Zones")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Sales")
    ax.xaxis.set_major_formatter(lambda x, _: money(x))
    ax.yaxis.set_major_formatter(lambda y, _: money(y))
    ax.legend(title="Margin Flag")

    counts = df["Margin_Flag"].value_counts()
    insight_1 = f"High-margin rows dominate the data with {counts.get('High', 0):,} records."
    insight_2 = f"Low-margin rows account for {counts.get('Low', 0):,} records and should be reviewed for pricing risk."
    save_chart(fig, "08_cost_vs_sales_margin_zones.png", insight_1, insight_2)
    print_chart_insight("Chart 8 - Cost vs Sales with Margin Zones", insight_1, insight_2)


def print_eda_summary(df: pd.DataFrame, product: pd.DataFrame) -> None:
    total_revenue = df["Sales"].sum()
    total_gross_profit = df["Gross Profit"].sum()
    company_margin = (total_gross_profit / total_revenue) * 100

    division_margin = (
        df.groupby("Division")
        .apply(lambda x: weighted_margin(x["Gross Profit"], x["Sales"]))
        .sort_values(ascending=False)
    )
    product_margin_flags = product.copy()
    product_margin_flags["Margin_Flag"] = pd.cut(
        product_margin_flags["Gross_Margin_Pct"],
        bins=[-np.inf, 20, 40, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype("object")
    risk_products = product_margin_flags[product_margin_flags["Gross_Margin_Pct"] < 20].sort_values("Gross_Margin_Pct")
    top_ppu = product.nlargest(3, "Profit_Per_Unit")
    bottom_ppu = product.nsmallest(3, "Profit_Per_Unit")

    print("\nStructured EDA Summary")
    print(f"Total revenue: {money(total_revenue)}")
    print(f"Total gross profit: {money(total_gross_profit)}")
    print(f"Company-wide margin %: {pct(company_margin)}")
    print(f"Division with highest margin: {division_margin.index[0]} ({pct(division_margin.iloc[0])})")
    print(f"Division with lowest margin: {division_margin.index[-1]} ({pct(division_margin.iloc[-1])})")
    print("\nProducts in each Margin_Flag category:")
    print(product_margin_flags["Margin_Flag"].value_counts().to_string())
    print("\nProducts flagged as margin risk (Gross_Margin_Pct < 20%):")
    if risk_products.empty:
        print("None")
    else:
        print(risk_products[["Product Name", "Division", "Gross_Margin_Pct"]].to_string(index=False))
    print("\nTop 3 products by Profit_Per_Unit:")
    print(top_ppu[["Product Name", "Division", "Profit_Per_Unit"]].to_string(index=False))
    print("\nBottom 3 products by Profit_Per_Unit:")
    print(bottom_ppu[["Product Name", "Division", "Profit_Per_Unit"]].to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    product = build_product_summary(df)

    sns.set_theme(style="whitegrid")
    chart_1_revenue_by_division(df)
    chart_2_margin_by_division(df)
    chart_3_revenue_vs_profit(product)
    chart_4_margin_distribution(df)
    chart_5_top_bottom_margin_products(product)
    chart_6_quarterly_revenue(df)
    chart_7_pareto_profit(product)
    chart_8_cost_vs_sales(df)
    print_eda_summary(df, product)
    print(f"\nCharts saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
