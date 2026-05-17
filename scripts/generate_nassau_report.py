from __future__ import annotations

from datetime import date
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
from fpdf import FPDF


INPUT_PATH = Path("data/processed/nassau_cleaned.csv")
OUTPUT_PATH = Path("reports/nassau_candy_report.pdf")

BROWN = (123, 63, 0)
PINK = (255, 105, 180)
GRAY = (105, 105, 105)
LIGHT_GRAY = (242, 242, 242)
RED = (198, 40, 40)
GREEN = (46, 125, 50)
BLACK = (30, 30, 30)


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:,.2f}%"


def weighted_margin(profit: pd.Series, sales: pd.Series) -> float:
    total_sales = sales.sum()
    return 0.0 if total_sales == 0 else (profit.sum() / total_sales) * 100


def margin_flag(value: float) -> str:
    if value >= 40:
        return "High"
    if value >= 20:
        return "Medium"
    return "Low"


def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
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
    product["Weighted_Gross_Margin_Pct"] = product["Total_Gross_Profit"] / product["Total_Sales"] * 100
    product["Profit_Per_Unit"] = product["Total_Gross_Profit"] / product["Total_Units"]
    product["Revenue_Contribution_Pct"] = product["Total_Sales"] / product["Total_Sales"].sum() * 100
    product["Profit_Contribution_Pct"] = product["Total_Gross_Profit"] / product["Total_Gross_Profit"].sum() * 100
    product["Margin_Flag"] = product["Weighted_Gross_Margin_Pct"].apply(margin_flag)
    product["Cost_To_Revenue_Ratio"] = product["Total_Cost"] / product["Total_Sales"]

    sales_median = product["Total_Sales"].median()
    margin_median = product["Weighted_Gross_Margin_Pct"].median()
    high_sales = product["Total_Sales"] >= sales_median
    high_margin = product["Weighted_Gross_Margin_Pct"] >= margin_median
    product["Quadrant"] = np.select(
        [high_sales & high_margin, high_sales & ~high_margin, ~high_sales & high_margin, ~high_sales & ~high_margin],
        ["Star", "Cash Drain", "Niche", "Deadweight"],
        default="Unclassified",
    )
    product["Cost_Flag"] = np.where(product["Cost_To_Revenue_Ratio"] > 0.75, "Cost Heavy", "Normal")
    product["Pricing_Flag"] = np.where(product["Weighted_Gross_Margin_Pct"] < 15, "Pricing Inefficiency", "Normal")
    product["Discontinuation_Flag"] = np.where(
        (product["Margin_Flag"] == "Low") & (product["Profit_Contribution_Pct"] < 1),
        "Discontinuation Candidate",
        "Retain",
    )
    return product.sort_values("Total_Gross_Profit", ascending=False)


def build_division_summary(df: pd.DataFrame) -> pd.DataFrame:
    division = (
        df.groupby("Division", as_index=False)
        .agg(
            Revenue=("Sales", "sum"),
            Profit=("Gross Profit", "sum"),
            Avg_Margin_Pct=("Gross_Margin_Pct", "mean"),
        )
    )
    division["Weighted_Margin_Pct"] = division["Profit"] / division["Revenue"] * 100
    division["Revenue_Share_Pct"] = division["Revenue"] / division["Revenue"].sum() * 100
    division["Profit_Share_Pct"] = division["Profit"] / division["Profit"].sum() * 100
    division["Revenue_to_Profit_Ratio"] = division["Revenue"] / division["Profit"]
    division["Efficiency_Flag"] = np.where(division["Revenue_to_Profit_Ratio"] > 4, "Margin inefficient", "Efficient")
    return division.sort_values("Profit", ascending=False)


def build_factory_summary(df: pd.DataFrame) -> pd.DataFrame:
    factory = (
        df.groupby("Factory", as_index=False)
        .agg(
            Profit=("Gross Profit", "sum"),
            Sales=("Sales", "sum"),
            Avg_Margin_Pct=("Gross_Margin_Pct", "mean"),
            SKU_Count=("Product Name", "nunique"),
        )
    )
    factory["Weighted_Margin_Pct"] = factory["Profit"] / factory["Sales"] * 100
    return factory.sort_values("Profit", ascending=False)


class ReportPDF(FPDF):
    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Nassau Candy Distributor | Internal Analytics | Confidential | Page {self.page_no()}", align="C")

    def chapter_title(self, title: str) -> None:
        self.set_fill_color(*BROWN)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)
        self.set_text_color(*BLACK)

    def paragraph(self, text: str, line_height: float = 5.2) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        self.set_x(self.l_margin)
        self.multi_cell(0, line_height, text)
        self.ln(2)

    def bullet_list(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 10)
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.2, f"- {item}")
        self.ln(2)

    def numbered_list(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 10)
        for index, item in enumerate(items, 1):
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.2, f"{index}. {item}")
        self.ln(2)

    def simple_table(self, headers: list[str], rows: list[list[str]], widths: list[float], lowest_index: int | None = None) -> None:
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*LIGHT_GRAY)
        self.set_text_color(*BLACK)
        for header, width in zip(headers, widths):
            self.cell(width, 7, header, border=1, fill=True)
        self.ln()

        self.set_font("Helvetica", "", 7.5)
        for row_index, row in enumerate(rows):
            if lowest_index is not None and row_index == lowest_index:
                self.set_fill_color(255, 230, 230)
                fill = True
            else:
                self.set_fill_color(255, 255, 255)
                fill = False
            max_lines = max(1, max(len(wrap(str(cell), width=max(int(width / 2.2), 8))) for cell, width in zip(row, widths)))
            row_height = 4.5 * max_lines
            if self.get_y() + row_height > self.page_break_trigger:
                self.add_page()
                self.set_font("Helvetica", "B", 8)
                self.set_fill_color(*LIGHT_GRAY)
                for header, width in zip(headers, widths):
                    self.cell(width, 7, header, border=1, fill=True)
                self.ln()
                self.set_font("Helvetica", "", 7.5)
            x_start = self.get_x()
            y_start = self.get_y()
            for cell, width in zip(row, widths):
                x = self.get_x()
                y = self.get_y()
                self.multi_cell(width, 4.5, str(cell), border=1, fill=fill)
                self.set_xy(x + width, y)
            self.set_xy(x_start, y_start + row_height)
        self.ln(4)


def table_rows(df: pd.DataFrame, columns: list[str], formatters: dict[str, callable] | None = None) -> list[list[str]]:
    formatters = formatters or {}
    rows = []
    for _, row in df.iterrows():
        rows.append([formatters.get(col, str)(row[col]) for col in columns])
    return rows


def add_cover(pdf: ReportPDF) -> None:
    pdf.add_page()
    pdf.set_fill_color(*BROWN)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_y(58)
    pdf.set_text_color(*BROWN)
    pdf.set_font("Helvetica", "B", 24)
    pdf.multi_cell(0, 12, "Product Line Profitability & Margin Performance Analysis", align="C")
    pdf.ln(6)
    pdf.set_text_color(*BLACK)
    pdf.set_font("Helvetica", "", 15)
    pdf.multi_cell(0, 8, "Nassau Candy Distributor - Internal Analytics Report", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Date: {date.today().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "For Internal Use Only", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(250)
    pdf.set_text_color(*GRAY)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "Prepared for Nassau Candy Distributor leadership", align="C")


def generate_report() -> None:
    df = pd.read_csv(INPUT_PATH)
    product = build_product_summary(df)
    division = build_division_summary(df)
    factory = build_factory_summary(df)

    total_revenue = df["Sales"].sum()
    total_profit = df["Gross Profit"].sum()
    company_margin = weighted_margin(df["Gross Profit"], df["Sales"])
    sku_count = df["Product Name"].nunique()
    top5_profit = product.nlargest(5, "Total_Gross_Profit")
    bottom5_margin = product.nsmallest(5, "Weighted_Gross_Margin_Pct")
    quadrant_counts = product["Quadrant"].value_counts().reindex(["Star", "Cash Drain", "Niche", "Deadweight"], fill_value=0)
    discontinuation = product[product["Discontinuation_Flag"] == "Discontinuation Candidate"].copy()
    pareto_profit = product.sort_values("Total_Gross_Profit", ascending=False).copy()
    pareto_profit["Cumulative_Profit_Pct"] = pareto_profit["Total_Gross_Profit"].cumsum() / pareto_profit["Total_Gross_Profit"].sum() * 100
    products_for_80_profit = int((pareto_profit["Cumulative_Profit_Pct"] < 80).sum() + 1)
    product_dependency = product[product["Profit_Contribution_Pct"] > 30]
    division_dependency = division[division["Profit_Share_Pct"] > 60]
    cost_heavy = product[product["Cost_Flag"] == "Cost Heavy"]
    pricing_inefficient = product[product["Pricing_Flag"] == "Pricing Inefficiency"]

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    add_cover(pdf)

    # Section 1
    pdf.add_page()
    pdf.chapter_title("SECTION 1 - Executive Summary")
    pdf.bullet_list(
        [
            f"Total revenue is {money(total_revenue)} with total gross profit of {money(total_profit)} and a company-wide weighted margin of {pct(company_margin)}.",
            f"The top {products_for_80_profit} products, representing {products_for_80_profit / sku_count * 100:.1f}% of SKUs, generate at least 80% of total profit.",
            f"Chocolate contributes {pct(float(division.loc[division['Division'].eq('Chocolate'), 'Profit_Share_Pct'].iloc[0]))} of gross profit, creating division-level concentration risk.",
            f"{product.iloc[0]['Product Name']} is the highest profit product at {money(product.iloc[0]['Total_Gross_Profit'])}.",
            f"{', '.join(discontinuation['Product Name']) if not discontinuation.empty else 'No products'} are flagged for discontinuation review based on low margin and low profit contribution.",
        ]
    )
    pdf.simple_table(
        ["KPI", "Value"],
        [
            ["Total Revenue", money(total_revenue)],
            ["Total Gross Profit", money(total_profit)],
            ["Avg Margin %", pct(company_margin)],
            ["SKU Count", f"{sku_count:,}"],
        ],
        [75, 75],
    )
    pdf.paragraph(
        "The portfolio is profitable overall, but leadership should not manage it through sales volume alone. "
        "Profit concentration in Chocolate masks operational weakness in selected Other-division SKUs, especially products with high cost-to-revenue ratios. "
        "The immediate management opportunity is to protect the leading Wonka Bar economics while repricing, renegotiating cost, or retiring structurally weak SKUs."
    )

    # Section 2
    pdf.add_page()
    pdf.chapter_title("SECTION 2 - Background & Problem Statement")
    pdf.paragraph(
        "Sales volume alone is misleading for Nassau Candy because high-revenue products can still dilute earnings when cost structures are unfavorable. "
        "Revenue also obscures concentration risk: a product line can appear healthy while a small number of products or one division carries nearly all profit."
    )
    pdf.bullet_list(
        [
            "Which products create the largest gross profit pool rather than just the largest sales pool.",
            "Which products have weak margin economics and should be repriced, renegotiated, or reviewed for discontinuation.",
            "How profit is concentrated across products, divisions, and factories.",
            "Whether factory and division performance is efficient enough to support growth plans.",
        ]
    )
    pdf.paragraph(
        "The analysis covers the three active product divisions: Chocolate, Sugar, and Other. "
        "It also evaluates the five factories present in the cleaned data: Lot's O' Nuts, Wicked Choccy's, Secret Factory, The Other Factory, and Sugar Shack."
    )

    # Section 3
    pdf.add_page()
    pdf.chapter_title("SECTION 3 - Methodology")
    pdf.bullet_list(
        [
            "Loaded the source transaction file and profiled shape, data types, and null counts.",
            "Validated Sales, Cost, Units, and Gross Profit; removed invalid non-positive Sales or Units records.",
            "Imputed numeric missing values with medians and removed rows missing required product or division labels.",
            "Removed exact duplicates, parsed dates, standardized division labels, and normalized product names.",
            "Mapped each product and division to its factory and calculated profitability KPI fields.",
        ]
    )
    pdf.paragraph(
        "KPI formulas used: Gross Margin % = Gross Profit / Sales x 100; "
        "Profit Per Unit = Gross Profit / Units; "
        "Revenue Contribution % = Product Sales / Total Sales x 100; "
        "Cost-to-Revenue Ratio = Cost / Sales."
    )
    pdf.paragraph(
        "Validation included gross profit reconciliation against Sales minus Cost with a $0.01 tolerance, post-cleaning null checks, duplicate checks, and factory mapping coverage checks."
    )

    # Section 4
    pdf.add_page()
    pdf.chapter_title("SECTION 4 - Product-Level Findings")
    pdf.paragraph("Top 5 products by gross profit:")
    pdf.simple_table(
        ["Product", "Division", "Sales", "Gross Profit", "Margin"],
        table_rows(
            top5_profit,
            ["Product Name", "Division", "Total_Sales", "Total_Gross_Profit", "Weighted_Gross_Margin_Pct"],
            {"Total_Sales": money, "Total_Gross_Profit": money, "Weighted_Gross_Margin_Pct": pct},
        ),
        [58, 25, 31, 34, 22],
    )
    pdf.paragraph("Bottom 5 products by margin percentage:")
    pdf.simple_table(
        ["Product", "Division", "Sales", "Gross Profit", "Margin"],
        table_rows(
            bottom5_margin,
            ["Product Name", "Division", "Total_Sales", "Total_Gross_Profit", "Weighted_Gross_Margin_Pct"],
            {"Total_Sales": money, "Total_Gross_Profit": money, "Weighted_Gross_Margin_Pct": pct},
        ),
        [58, 25, 31, 34, 22],
    )
    pdf.simple_table(
        ["Quadrant", "SKU Count"],
        [[index, f"{value:,}"] for index, value in quadrant_counts.items()],
        [70, 35],
    )
    pdf.paragraph("Products flagged for discontinuation review:")
    if discontinuation.empty:
        pdf.paragraph("No products meet the discontinuation criteria.")
    else:
        pdf.simple_table(
            ["Product", "Margin", "Profit Share", "Cost Ratio"],
            table_rows(
                discontinuation,
                ["Product Name", "Weighted_Gross_Margin_Pct", "Profit_Contribution_Pct", "Cost_To_Revenue_Ratio"],
                {
                    "Weighted_Gross_Margin_Pct": pct,
                    "Profit_Contribution_Pct": pct,
                    "Cost_To_Revenue_Ratio": lambda x: pct(x * 100),
                },
            ),
            [65, 30, 34, 30],
        )

    # Section 5
    pdf.add_page()
    pdf.chapter_title("SECTION 5 - Division Performance Analysis")
    lowest_division_margin_idx = int(division["Weighted_Margin_Pct"].reset_index(drop=True).idxmin())
    pdf.simple_table(
        ["Division", "Revenue", "Profit", "Margin", "Profit Share", "Flag"],
        table_rows(
            division,
            ["Division", "Revenue", "Profit", "Weighted_Margin_Pct", "Profit_Share_Pct", "Efficiency_Flag"],
            {"Revenue": money, "Profit": money, "Weighted_Margin_Pct": pct, "Profit_Share_Pct": pct},
        ),
        [28, 34, 34, 24, 28, 36],
        lowest_index=lowest_division_margin_idx,
    )
    lowest_factory_margin_idx = int(factory["Weighted_Margin_Pct"].reset_index(drop=True).idxmin())
    pdf.simple_table(
        ["Factory", "Profit", "Margin", "SKU Count"],
        table_rows(
            factory,
            ["Factory", "Profit", "Weighted_Margin_Pct", "SKU_Count"],
            {"Profit": money, "Weighted_Margin_Pct": pct, "SKU_Count": lambda x: f"{int(x):,}"},
        ),
        [58, 38, 30, 28],
        lowest_index=lowest_factory_margin_idx,
    )
    for _, row in division.sort_values("Division").iterrows():
        pdf.paragraph(
            f"{row['Division']}: Revenue is {money(row['Revenue'])}, gross profit is {money(row['Profit'])}, "
            f"and weighted margin is {pct(row['Weighted_Margin_Pct'])}. "
            f"The division contributes {pct(row['Profit_Share_Pct'])} of profit and is classified as {row['Efficiency_Flag'].lower()}."
        )

    # Section 6
    pdf.add_page()
    pdf.chapter_title("SECTION 6 - Pareto & Concentration Risk")
    pdf.paragraph(
        f"{products_for_80_profit} products ({products_for_80_profit / sku_count * 100:.1f}% of SKUs) generate 80% of total profit. "
        f"The first product crossing the 80% threshold is {pareto_profit.iloc[products_for_80_profit - 1]['Product Name']}."
    )
    pdf.bullet_list(
        [
            f"Single-product dependency risk above 30% of profit: {', '.join(product_dependency['Product Name']) if not product_dependency.empty else 'None'}.",
            f"Division dependency risk above 60% of profit: {', '.join(division_dependency['Division']) if not division_dependency.empty else 'None'}.",
            "Reduce concentration risk by building attach strategies for Sugar and Other products with strong margins but low volume.",
            "Protect the leading Chocolate portfolio through price discipline and supply continuity planning.",
        ]
    )

    # Section 7
    pdf.add_page()
    pdf.chapter_title("SECTION 7 - Cost Structure Diagnostics")
    pdf.paragraph(f"Cost-heavy products: {', '.join(cost_heavy['Product Name']) if not cost_heavy.empty else 'None'}.")
    pdf.paragraph(f"Pricing inefficiency flags: {', '.join(pricing_inefficient['Product Name']) if not pricing_inefficient.empty else 'None'}.")
    repricing = product[(product["Margin_Flag"].isin(["Low", "Medium"])) | (product["Weighted_Gross_Margin_Pct"] < company_margin)].nsmallest(5, "Weighted_Gross_Margin_Pct")
    renegotiation = product.nlargest(5, "Cost_To_Revenue_Ratio")
    discontinue = discontinuation
    pdf.simple_table(
        ["Action", "Recommended SKUs"],
        [
            ["Repricing", ", ".join(repricing["Product Name"])],
            ["Cost renegotiation", ", ".join(renegotiation["Product Name"])],
            ["Discontinuation", ", ".join(discontinue["Product Name"]) if not discontinue.empty else "None"],
        ],
        [45, 135],
    )

    # Section 8
    pdf.add_page()
    pdf.chapter_title("SECTION 8 - Recommendations")
    pdf.paragraph("Immediate actions - this quarter:")
    pdf.numbered_list(
        [
            f"Open a discontinuation and pricing review for {', '.join(discontinuation['Product Name']) if not discontinuation.empty else 'all low-margin candidates'}; expected impact: remove or repair structurally weak margin drag.",
            f"Renegotiate cost inputs for {', '.join(renegotiation.head(3)['Product Name'])}; expected impact: lower cost-to-revenue exposure in the weakest SKUs.",
            "Protect Chocolate pricing and inventory availability; expected impact: preserve the division currently funding most company profit.",
        ]
    )
    pdf.paragraph("Medium-term initiatives - 6 months:")
    pdf.numbered_list(
        [
            "Bundle high-margin, low-volume Sugar and Other SKUs with leading Chocolate products; expected impact: diversify profit without sacrificing margin quality.",
            "Build monthly product profitability monitoring with alerts for margin below 20% or cost ratio above 75%; expected impact: faster intervention.",
            "Review factory-level production economics for The Other Factory; expected impact: identify supplier or production changes for low-margin output.",
        ]
    )
    pdf.paragraph("Strategic review items - annual:")
    pdf.numbered_list(
        [
            "Set target profit contribution by division; expected impact: reduce over-dependency on Chocolate.",
            "Conduct SKU rationalization using quadrant status and Pareto contribution; expected impact: focus management attention on profit-relevant lines.",
            "Evaluate whether low-volume niche SKUs support strategic assortment goals; expected impact: clarify which products deserve investment despite small revenue.",
        ]
    )

    # Section 9
    pdf.add_page()
    pdf.chapter_title("SECTION 9 - Conclusion")
    pdf.paragraph(
        "Nassau Candy Distributor has a profitable portfolio, but the economics are uneven. "
        "Chocolate products carry most of the profit base, while several smaller SKUs contribute little to enterprise earnings. "
        "The company should therefore manage profitability at the product and factory level, not only at the division revenue level."
    )
    pdf.paragraph(
        "The analysis shows that a small number of products drive the majority of profit. "
        "This is operationally efficient in the short run, but it creates concentration exposure if demand, supplier cost, or pricing power changes in the leading Chocolate line."
    )
    pdf.paragraph(
        "Leadership should prioritize margin repair on weak SKUs, protect high-profit Chocolate products, and develop a more balanced growth path through selected Sugar and Other products. "
        "The recommended next step is to convert this report into a recurring monthly profitability review with action owners for pricing, sourcing, and product portfolio decisions."
    )
    pdf.paragraph(
        "Closing statement: Nassau Candy can improve resilience and profitability by shifting from sales-led reporting to margin-led product management."
    )

    pdf.output(OUTPUT_PATH)


if __name__ == "__main__":
    generate_report()
