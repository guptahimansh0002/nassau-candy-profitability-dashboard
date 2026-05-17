from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/nassau_cleaned.csv")


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:,.2f}%"


def ratio(value: float) -> str:
    return f"{value:,.2f}"


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    if df.empty:
        print("No rows to display.")
    else:
        print(df.to_string(index=False))


def weighted_margin(profit: pd.Series, sales: pd.Series) -> float:
    total_sales = sales.sum()
    if total_sales == 0:
        return 0.0
    return (profit.sum() / total_sales) * 100


def format_financial_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    money_columns = [
        "Total Sales",
        "Total Cost",
        "Total Gross Profit",
        "Total Revenue",
        "Avg Profit_Per_Unit",
    ]
    pct_columns = [
        "Avg Gross_Margin_Pct",
        "Weighted Gross_Margin_Pct",
        "Revenue_Contribution_Pct",
        "Profit_Contribution_Pct",
        "Revenue Share %",
        "Profit Share %",
        "Avg Margin %",
        "Cost_To_Revenue_Ratio",
        "Cumulative Profit Contribution %",
        "Cumulative Revenue Contribution %",
    ]

    for column in money_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(money)
    for column in pct_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(pct)
    if "Revenue-to-Profit Ratio" in formatted.columns:
        formatted["Revenue-to-Profit Ratio"] = formatted["Revenue-to-Profit Ratio"].map(ratio)
    return formatted


def add_lowest_margin_highlight(df: pd.DataFrame, margin_column: str) -> pd.DataFrame:
    highlighted = df.copy()
    lowest_margin = highlighted[margin_column].min()
    highlighted["Lowest Margin Row"] = np.where(
        highlighted[margin_column].eq(lowest_margin),
        "<< LOWEST MARGIN",
        "",
    )
    return highlighted


def build_product_profitability(df: pd.DataFrame) -> pd.DataFrame:
    product = (
        df.groupby("Product Name", as_index=False)
        .agg(
            Division=("Division", lambda x: x.mode().iat[0]),
            Factory=("Factory", lambda x: x.mode().iat[0]),
            Total_Sales=("Sales", "sum"),
            Total_Cost=("Cost", "sum"),
            Total_Gross_Profit=("Gross Profit", "sum"),
            Total_Units=("Units", "sum"),
            Avg_Gross_Margin_Pct=("Gross_Margin_Pct", "mean"),
            Avg_Profit_Per_Unit=("Profit_Per_Unit", "mean"),
        )
    )
    total_sales = product["Total_Sales"].sum()
    total_profit = product["Total_Gross_Profit"].sum()
    product["Weighted_Gross_Margin_Pct"] = (
        product["Total_Gross_Profit"] / product["Total_Sales"] * 100
    )
    product["Revenue_Contribution_Pct"] = product["Total_Sales"] / total_sales * 100
    product["Profit_Contribution_Pct"] = product["Total_Gross_Profit"] / total_profit * 100
    product["Profit_Rank"] = product["Total_Gross_Profit"].rank(method="dense", ascending=False).astype(int)
    product["Margin_Rank"] = product["Weighted_Gross_Margin_Pct"].rank(method="dense", ascending=False).astype(int)

    sales_median = product["Total_Sales"].median()
    margin_median = product["Weighted_Gross_Margin_Pct"].median()
    high_sales = product["Total_Sales"] >= sales_median
    high_margin = product["Weighted_Gross_Margin_Pct"] >= margin_median
    product["Profitability Quadrant"] = np.select(
        [
            high_sales & high_margin,
            high_sales & ~high_margin,
            ~high_sales & high_margin,
            ~high_sales & ~high_margin,
        ],
        ["Star", "Cash Drain", "Niche", "Deadweight"],
        default="Unclassified",
    )

    return product.sort_values("Total_Gross_Profit", ascending=False)


def build_division_performance(df: pd.DataFrame) -> pd.DataFrame:
    division = (
        df.groupby("Division", as_index=False)
        .agg(
            Total_Revenue=("Sales", "sum"),
            Total_Gross_Profit=("Gross Profit", "sum"),
            Avg_Gross_Margin_Pct=("Gross_Margin_Pct", "mean"),
        )
    )
    division["Weighted_Gross_Margin_Pct"] = (
        division["Total_Gross_Profit"] / division["Total_Revenue"] * 100
    )
    division["Revenue Share %"] = division["Total_Revenue"] / division["Total_Revenue"].sum() * 100
    division["Profit Share %"] = division["Total_Gross_Profit"] / division["Total_Gross_Profit"].sum() * 100
    division["Revenue-to-Profit Ratio"] = division["Total_Revenue"] / division["Total_Gross_Profit"]
    division["Efficiency Flag"] = np.where(
        division["Revenue-to-Profit Ratio"] > 4,
        "margin inefficient",
        "efficient",
    )
    return division.sort_values("Total_Gross_Profit", ascending=False)


def build_factory_performance(df: pd.DataFrame) -> pd.DataFrame:
    factory = (
        df.groupby("Factory", as_index=False)
        .agg(
            Total_Gross_Profit=("Gross Profit", "sum"),
            Total_Sales=("Sales", "sum"),
            Avg_Margin_Pct=("Gross_Margin_Pct", "mean"),
            Number_of_SKUs=("Product Name", "nunique"),
        )
    )
    factory["Weighted_Gross_Margin_Pct"] = factory["Total_Gross_Profit"] / factory["Total_Sales"] * 100
    return factory.sort_values("Total_Gross_Profit", ascending=False)


def build_pareto(product: pd.DataFrame) -> pd.DataFrame:
    pareto = product.sort_values("Total_Gross_Profit", ascending=False).copy()
    pareto["Cumulative Profit Contribution %"] = (
        pareto["Total_Gross_Profit"].cumsum() / pareto["Total_Gross_Profit"].sum() * 100
    )
    revenue_sorted = product.sort_values("Total_Sales", ascending=False).copy()
    revenue_sorted["Cumulative Revenue Contribution %"] = (
        revenue_sorted["Total_Sales"].cumsum() / revenue_sorted["Total_Sales"].sum() * 100
    )
    return pareto, revenue_sorted


def build_diagnostics(product: pd.DataFrame) -> pd.DataFrame:
    diagnostics = product.copy()
    diagnostics["Cost_To_Revenue_Ratio"] = diagnostics["Total_Cost"] / diagnostics["Total_Sales"] * 100
    diagnostics["Cost Structure Flag"] = np.where(
        diagnostics["Cost_To_Revenue_Ratio"] > 75,
        "Cost Heavy",
        "Normal",
    )
    diagnostics["Pricing Flag"] = np.where(
        diagnostics["Weighted_Gross_Margin_Pct"] < 15,
        "Pricing Inefficiency",
        "Normal",
    )
    diagnostics["Margin_Flag"] = pd.cut(
        diagnostics["Weighted_Gross_Margin_Pct"],
        bins=[-np.inf, 20, 40, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)
    diagnostics["Discontinuation Flag"] = np.where(
        (diagnostics["Margin_Flag"] == "Low")
        & (diagnostics["Profit_Contribution_Pct"] < 1),
        "Discontinuation Candidate",
        "Retain",
    )
    return diagnostics.sort_values("Cost_To_Revenue_Ratio", ascending=False)


def print_section_summaries(
    product: pd.DataFrame,
    division: pd.DataFrame,
    factory: pd.DataFrame,
    pareto_profit: pd.DataFrame,
    pareto_revenue: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    total_skus = len(product)
    profit_80_count = int((pareto_profit["Cumulative Profit Contribution %"] < 80).sum() + 1)
    revenue_80_count = int((pareto_revenue["Cumulative Revenue Contribution %"] < 80).sum() + 1)
    highest_profit_product = product.iloc[0]
    highest_margin_product = product.sort_values("Weighted_Gross_Margin_Pct", ascending=False).iloc[0]
    lowest_margin_product = product.sort_values("Weighted_Gross_Margin_Pct").iloc[0]
    highest_margin_division = division.sort_values("Weighted_Gross_Margin_Pct", ascending=False).iloc[0]
    lowest_margin_division = division.sort_values("Weighted_Gross_Margin_Pct").iloc[0]
    top_factory = factory.iloc[0]
    cost_heavy = diagnostics[diagnostics["Cost Structure Flag"] == "Cost Heavy"]
    pricing_issues = diagnostics[diagnostics["Pricing Flag"] == "Pricing Inefficiency"]
    discontinue = diagnostics[diagnostics["Discontinuation Flag"] == "Discontinuation Candidate"]

    print("\n" + "=" * 100)
    print("Plain-English Summary of Findings")
    print("=" * 100)
    print("\nSECTION A: Product-Level Profitability")
    print(f"- {highest_profit_product['Product Name']} is the top profit generator with {money(highest_profit_product['Total_Gross_Profit'])} in gross profit.")
    print(f"- {highest_margin_product['Product Name']} has the strongest weighted margin at {pct(highest_margin_product['Weighted_Gross_Margin_Pct'])}.")
    print(f"- {lowest_margin_product['Product Name']} has the weakest weighted margin at {pct(lowest_margin_product['Weighted_Gross_Margin_Pct'])}.")
    print(f"- Quadrant mix: {product['Profitability Quadrant'].value_counts().to_dict()}.")

    print("\nSECTION B: Division and Factory Performance")
    print(f"- {highest_margin_division['Division']} has the highest division margin at {pct(highest_margin_division['Weighted_Gross_Margin_Pct'])}.")
    print(f"- {lowest_margin_division['Division']} has the lowest division margin at {pct(lowest_margin_division['Weighted_Gross_Margin_Pct'])}.")
    print(f"- {top_factory['Factory']} produces the most gross profit at {money(top_factory['Total_Gross_Profit'])}.")
    print(f"- Margin inefficient divisions flagged: {division[division['Efficiency Flag'].eq('margin inefficient')]['Division'].tolist() or 'None'}.")

    print("\nSECTION C: Pareto and Concentration")
    print(f"- {profit_80_count} products ({profit_80_count / total_skus * 100:.1f}% of SKUs) generate 80% of profit.")
    print(f"- {revenue_80_count} products ({revenue_80_count / total_skus * 100:.1f}% of SKUs) generate 80% of revenue.")
    print(f"- Single-product profit concentration above 30%: {product[product['Profit_Contribution_Pct'] > 30]['Product Name'].tolist() or 'None'}.")
    print(f"- Division profit concentration above 60%: {division[division['Profit Share %'] > 60]['Division'].tolist() or 'None'}.")

    print("\nSECTION D: Cost Structure Diagnostics")
    print(f"- Cost-heavy products: {cost_heavy['Product Name'].tolist() or 'None'}.")
    print(f"- Pricing inefficiency products: {pricing_issues['Product Name'].tolist() or 'None'}.")
    print(f"- Discontinuation candidates: {discontinue['Product Name'].tolist() or 'None'}.")
    print(f"- The highest cost-to-revenue ratio is {pct(diagnostics.iloc[0]['Cost_To_Revenue_Ratio'])} for {diagnostics.iloc[0]['Product Name']}.")


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)

    df = pd.read_csv(INPUT_PATH)
    product = build_product_profitability(df)
    division = build_division_performance(df)
    factory = build_factory_performance(df)
    pareto_profit, pareto_revenue = build_pareto(product)
    diagnostics = build_diagnostics(product)

    product_display = product.rename(
        columns={
            "Total_Sales": "Total Sales",
            "Total_Cost": "Total Cost",
            "Total_Gross_Profit": "Total Gross Profit",
            "Total_Units": "Total Units",
            "Avg_Gross_Margin_Pct": "Avg Gross_Margin_Pct",
            "Avg_Profit_Per_Unit": "Avg Profit_Per_Unit",
            "Weighted_Gross_Margin_Pct": "Weighted Gross_Margin_Pct",
        }
    )
    print_table(
        "SECTION A: Product-Level Profitability Ranking",
        format_financial_table(product_display),
    )

    division_display = division.rename(
        columns={
            "Total_Revenue": "Total Revenue",
            "Total_Gross_Profit": "Total Gross Profit",
            "Avg_Gross_Margin_Pct": "Avg Gross_Margin_Pct",
            "Weighted_Gross_Margin_Pct": "Weighted Gross_Margin_Pct",
        }
    )
    division_display = add_lowest_margin_highlight(division_display, "Weighted Gross_Margin_Pct")
    print_table(
        "SECTION B1: Division-Level Performance",
        format_financial_table(division_display),
    )

    factory_display = factory.rename(
        columns={
            "Total_Gross_Profit": "Total Gross Profit",
            "Total_Sales": "Total Sales",
            "Avg_Margin_Pct": "Avg Margin %",
            "Weighted_Gross_Margin_Pct": "Weighted Gross_Margin_Pct",
        }
    )
    factory_display = add_lowest_margin_highlight(factory_display, "Weighted Gross_Margin_Pct")
    print_table(
        "SECTION B2: Factory-Level Performance",
        format_financial_table(factory_display),
    )

    total_skus = len(product)
    profit_80_count = int((pareto_profit["Cumulative Profit Contribution %"] < 80).sum() + 1)
    revenue_80_count = int((pareto_revenue["Cumulative Revenue Contribution %"] < 80).sum() + 1)
    print("\n" + "=" * 100)
    print("SECTION C: Pareto & Concentration Analysis")
    print("=" * 100)
    print(f"{profit_80_count} products ({profit_80_count / total_skus * 100:.1f}% of SKUs) generate 80% of profit.")
    print(f"{revenue_80_count} products ({revenue_80_count / total_skus * 100:.1f}% of SKUs) generate 80% of revenue.")
    product_risk = product[product["Profit_Contribution_Pct"] > 30]
    division_risk = division[division["Profit Share %"] > 60]
    print(f"Single-product over-dependency risk (>30% profit): {product_risk['Product Name'].tolist() or 'None'}")
    print(f"Division over-dependency risk (>60% profit): {division_risk['Division'].tolist() or 'None'}")

    pareto_display = pareto_profit[
        [
            "Product Name",
            "Total_Gross_Profit",
            "Profit_Contribution_Pct",
            "Cumulative Profit Contribution %",
        ]
    ].rename(
        columns={
            "Total_Gross_Profit": "Total Gross Profit",
        }
    )
    print_table("SECTION C: Product Profit Pareto Table", format_financial_table(pareto_display))

    diagnostics_display = diagnostics.rename(
        columns={
            "Total_Sales": "Total Sales",
            "Total_Cost": "Total Cost",
            "Total_Gross_Profit": "Total Gross Profit",
            "Weighted_Gross_Margin_Pct": "Weighted Gross_Margin_Pct",
        }
    )
    diagnostics_display = diagnostics_display[
        [
            "Product Name",
            "Division",
            "Total Sales",
            "Total Cost",
            "Total Gross Profit",
            "Weighted Gross_Margin_Pct",
            "Profit_Contribution_Pct",
            "Cost_To_Revenue_Ratio",
            "Margin_Flag",
            "Cost Structure Flag",
            "Pricing Flag",
            "Discontinuation Flag",
        ]
    ]
    print_table("SECTION D: Cost Structure Diagnostics", format_financial_table(diagnostics_display))

    print_section_summaries(product, division, factory, pareto_profit, pareto_revenue, diagnostics)


if __name__ == "__main__":
    main()
