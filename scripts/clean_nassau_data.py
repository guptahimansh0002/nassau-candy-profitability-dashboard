from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/Nassau Candy Distributor.csv")
OUTPUT_PATH = Path("data/processed/nassau_cleaned.csv")

NUMERIC_COLUMNS = ["Sales", "Cost", "Units", "Gross Profit"]
REQUIRED_CATEGORY_COLUMNS = ["Division", "Product Name"]

DIVISION_VARIANTS = {
    "CHOC": "Chocolate",
    "CHOCOLATE": "Chocolate",
    "SUGAR": "Sugar",
    "OTHER": "Other",
}

FACTORY_MAPPING = {
    ("Chocolate", "Wonka Bar - Nutty Crunch Surprise"): "Lot's O' Nuts",
    ("Chocolate", "Wonka Bar - Fudge Mallows"): "Lot's O' Nuts",
    ("Chocolate", "Wonka Bar - Scrumdiddlyumptious"): "Lot's O' Nuts",
    ("Chocolate", "Wonka Bar - Milk Chocolate"): "Wicked Choccy's",
    ("Chocolate", "Wonka Bar - Triple Dazzle Caramel"): "Wicked Choccy's",
    ("Sugar", "Laffy Taffy"): "Sugar Shack",
    ("Sugar", "SweeTARTS"): "Sugar Shack",
    ("Sugar", "Nerds"): "Sugar Shack",
    ("Sugar", "Fun Dip"): "Sugar Shack",
    ("Sugar", "Everlasting Gobstopper"): "Secret Factory",
    ("Sugar", "Hair Toffee"): "The Other Factory",
    ("Other", "Fizzy Lifting Drinks"): "Sugar Shack",
    ("Other", "Lickable Wallpaper"): "Secret Factory",
    ("Other", "Wonka Gum"): "Secret Factory",
    ("Other", "Kazookles"): "The Other Factory",
}

PRODUCT_DIVISION_MAPPING = {
    product_name: division
    for division, product_name in FACTORY_MAPPING
}


def print_initial_profile(df: pd.DataFrame) -> None:
    print("Initial shape:")
    print(df.shape)
    print("\nInitial dtypes:")
    print(df.dtypes)
    print("\nInitial null counts:")
    print(df.isna().sum())


def standardize_division(value: object) -> object:
    if pd.isna(value):
        return value

    normalized = str(value).strip()
    lookup_key = normalized.upper()
    if lookup_key in DIVISION_VARIANTS:
        return DIVISION_VARIANTS[lookup_key]

    return normalized.title()


def standardize_product_name(value: object) -> object:
    if pd.isna(value):
        return value

    normalized = " ".join(str(value).strip().split())
    return normalized.replace("Wonka Bar -Scrumdiddlyumptious", "Wonka Bar - Scrumdiddlyumptious")


def add_factory(df: pd.DataFrame) -> pd.DataFrame:
    mapping_keys = list(zip(df["Division"], df["Product Name"]))
    df["Factory"] = [FACTORY_MAPPING.get(key, "Unmapped") for key in mapping_keys]
    return df


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    total_sales = df["Sales"].sum()
    total_gross_profit = df["Gross Profit"].sum()

    df["Gross_Margin_Pct"] = ((df["Gross Profit"] / df["Sales"]) * 100).round(2)
    df["Profit_Per_Unit"] = (df["Gross Profit"] / df["Units"]).round(2)
    df["Revenue_Contribution_Pct"] = np.where(
        total_sales != 0,
        (df["Sales"] / total_sales) * 100,
        0,
    )
    df["Profit_Contribution_Pct"] = np.where(
        total_gross_profit != 0,
        (df["Gross Profit"] / total_gross_profit) * 100,
        0,
    )

    df["Gross_Margin_Pct"] = df["Gross_Margin_Pct"].round(2)
    df["Revenue_Contribution_Pct"] = df["Revenue_Contribution_Pct"].round(2)
    df["Profit_Contribution_Pct"] = df["Profit_Contribution_Pct"].round(2)

    df["Margin_Flag"] = pd.cut(
        df["Gross_Margin_Pct"],
        bins=[-np.inf, 20, 40, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype("object")

    return df


def clean_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    initial_row_count = len(df)

    print_initial_profile(df)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_sales_units_mask = (df["Sales"] <= 0) | (df["Units"] <= 0)
    invalid_sales_units_count = int(invalid_sales_units_mask.sum())
    df = df.loc[~invalid_sales_units_mask].copy()

    for column in NUMERIC_COLUMNS:
        df[column] = df[column].fillna(df[column].median())

    missing_required_mask = df[REQUIRED_CATEGORY_COLUMNS].isna().any(axis=1)
    missing_required_count = int(missing_required_mask.sum())
    df = df.loc[~missing_required_mask].copy()

    before_duplicates = len(df)
    df = df.drop_duplicates().copy()
    duplicate_count = before_duplicates - len(df)

    df["Division"] = df["Division"].apply(standardize_division)
    df["Product Name"] = df["Product Name"].apply(standardize_product_name)
    expected_division = df["Product Name"].map(PRODUCT_DIVISION_MAPPING)
    division_correction_mask = expected_division.notna() & (df["Division"] != expected_division)
    division_correction_count = int(division_correction_mask.sum())
    df.loc[division_correction_mask, "Division"] = expected_division[division_correction_mask]

    df["Gross_Profit_Mismatch_Flag"] = ~np.isclose(
        df["Gross Profit"],
        df["Sales"] - df["Cost"],
        atol=0.01,
        rtol=0,
    )
    gross_profit_mismatch_count = int(df["Gross_Profit_Mismatch_Flag"].sum())

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

    df["Year"] = df["Order Date"].dt.year.astype("Int64")
    df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype("Int64").astype(str)
    df["Month"] = df["Order Date"].dt.month.astype("Int64")
    df["Month_Name"] = df["Order Date"].dt.month_name()

    df = add_factory(df)
    df = add_kpis(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal validation report:")
    print(f"Rows before cleaning: {initial_row_count}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Rows removed for Sales <= 0 or Units <= 0: {invalid_sales_units_count}")
    print(f"Rows removed for missing Division/Product Name: {missing_required_count}")
    print(f"Exact duplicate rows removed: {duplicate_count}")
    print(f"Product/division corrections applied: {division_correction_count}")
    print(f"Gross profit mismatch rows flagged: {gross_profit_mismatch_count}")
    print("\nFinal null summary:")
    print(df.isna().sum())
    print("\nMargin flag distribution:")
    print(df["Margin_Flag"].value_counts(dropna=False))
    print("\nDivision counts:")
    print(df["Division"].value_counts(dropna=False))
    print(f"\nCleaned data saved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    clean_data()
