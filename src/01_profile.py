"""Basic profiling of the Inside Airbnb Sicily listings dataset.

This script loads the raw listings dataset without modifying it and creates
a column-level profile containing data types, missing-value counts and
uniqueness information. It also reports basic row-level uniqueness checks.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(
        description="Create a basic profile of the Airbnb listings dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw listings CSV or CSV.GZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which profiling outputs will be stored.",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the raw listings dataset from disk."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    return pd.read_csv(input_path, low_memory=False)


def build_column_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create basic completeness and uniqueness statistics per column."""
    row_count = len(data)

    profile = pd.DataFrame(
        {
            "column": data.columns,
            "dtype": data.dtypes.astype(str).values,
            "non_null_count": data.notna().sum().values,
            "missing_count": data.isna().sum().values,
            "unique_non_null": data.nunique(dropna=True).values,
        }
    )

    profile["missing_percent"] = (
        profile["missing_count"] / row_count * 100
    ).round(3)

    return profile


def count_duplicate_rows(data: pd.DataFrame) -> int:
    """Return the number of exact duplicate rows."""
    return int(data.duplicated().sum())


def count_duplicate_listing_ids(data: pd.DataFrame) -> int | None:
    """Return the number of duplicated listing IDs, if the column exists."""
    if "id" not in data.columns:
        return None

    return int(data["id"].duplicated().sum())


def parse_price_values(data: pd.DataFrame) -> pd.Series:
    """Convert the raw price strings to numeric values for profiling."""
    if "price" not in data.columns:
        raise KeyError("Expected column 'price' was not found.")

    cleaned_price = (
        data["price"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    return pd.to_numeric(cleaned_price, errors="coerce")


def extract_quote_currency(value: object) -> str | None:
    """Extract the currency code from a raw Airbnb price quote.

    Missing quote values remain missing. Invalid JSON is marked explicitly
    so that malformed source data are distinguishable from ordinary
    missingness.
    """
    if pd.isna(value):
        return None

    try:
        parsed_value = json.loads(str(value))
    except json.JSONDecodeError:
        return "__PARSE_ERROR__"

    return parsed_value.get("quote", {}).get("currency")


def build_price_representation_checks(
    data: pd.DataFrame,
    price_numeric: pd.Series,
) -> pd.DataFrame:
    """Assess consistency between the available price representations.

    The dataset contains a formatted price string as well as numeric quote
    information. These representations are compared to determine whether
    the apparent currency symbol in the raw price field is merely a display
    format or reflects a genuine difference in the underlying values.
    """
    required_columns = {
        "price",
        "price_quote_raw",
        "price_quote_price_per_night",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing_names}")

    # A parse failure occurs only when an original price value exists but
    # cannot be converted to a numeric representation.
    raw_price_present = data["price"].notna()
    parse_failures = raw_price_present & price_numeric.isna()

    # Currency information is embedded in the JSON-formatted quote field
    # and therefore needs to be extracted before it can be assessed.
    quote_currency = data["price_quote_raw"].map(
        extract_quote_currency
    )

    # Distinguish entirely missing quote records from quote records that
    # exist but do not contain currency information. Treating both cases
    # simply as "missing currency" would conceal different source-data
    # conditions.
    quote_raw_missing = data["price_quote_raw"].isna()
    currency_missing_in_quote = (
            data["price_quote_raw"].notna()
            & quote_currency.isna()
    )

    # Compare numeric values only where both representations are available.
    paired_prices = (
        price_numeric.notna()
        & data["price_quote_price_per_night"].notna()
    )

    numeric_difference = (
        price_numeric[paired_prices]
        - data.loc[
            paired_prices,
            "price_quote_price_per_night",
        ]
    ).abs()

    # A small tolerance avoids treating insignificant floating-point
    # representation differences as genuine price mismatches.
    price_mismatch_tolerance = 0.001

    checks = [
        {
            "check": "non_missing_raw_price",
            "count": int(raw_price_present.sum()),
        },
        {
            "check": "price_parse_failures",
            "count": int(parse_failures.sum()),
        },
        {
            "check": "paired_numeric_prices",
            "count": int(paired_prices.sum()),
        },
        {
            "check": "numeric_price_mismatches",
            "count": int(
                (
                        numeric_difference
                        > price_mismatch_tolerance
                ).sum()),
        },
        {
            "check": "quote_currency_eur",
            "count": int((quote_currency == "EUR").sum()),
        },
        {
            "check": "quote_raw_missing",
            "count": int(quote_raw_missing.sum()),
        },
        {
            "check": "quote_currency_missing_in_present_quote",
            "count": int(currency_missing_in_quote.sum()),
        },
        {
            "check": "quote_currency_parse_errors",
            "count": int(
                (
                        quote_currency
                        == "__PARSE_ERROR__"
                ).sum()
            ),
        },
    ]

    return pd.DataFrame(checks)


def build_price_missingness_by_source(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise missing price values for each listing source."""
    required_columns = {"price", "source"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing_names}")

    price_missing = data["price"].isna()

    summary = (
        data.assign(price_missing=price_missing)
        .groupby("source", dropna=False)
        .agg(
            row_count=("price_missing", "size"),
            missing_price_count=("price_missing", "sum"),
        )
        .reset_index()
    )

    summary["missing_price_percent"] = (
        summary["missing_price_count"] / summary["row_count"] * 100
    ).round(3)

    return summary


def write_price_missingness(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write the price-missingness summary to a CSV file."""
    output_path = output_dir / "missing_price_by_source.csv"
    summary.to_csv(output_path, index=False)

    return output_path


def write_price_representation_checks(
    checks: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write price-representation checks to a CSV file."""
    output_path = output_dir / "price_representation_checks.csv"
    checks.to_csv(output_path, index=False)

    return output_path


def write_column_profile(
    profile: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write the column-level profile to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "column_profile.csv"

    profile.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    """Run the basic profiling workflow."""
    args = parse_arguments()

    data = load_dataset(args.input)
    column_profile = build_column_profile(data)
    price_numeric = parse_price_values(data)

    price_checks = build_price_representation_checks(
        data,
        price_numeric,
    )

    price_missingness = build_price_missingness_by_source(data)

    duplicate_rows = count_duplicate_rows(data)
    duplicate_ids = count_duplicate_listing_ids(data)

    output_path = write_column_profile(
        column_profile,
        args.output_dir,
    )

    price_output_path = write_price_missingness(
        price_missingness,
        args.output_dir,
    )

    price_checks_path = write_price_representation_checks(
        price_checks,
        args.output_dir,
    )

    print("Basic profiling complete.")
    print(f"Rows x columns: {len(data):,} x {len(data.columns):,}")
    print(f"Exact duplicate rows: {duplicate_rows:,}")

    if duplicate_ids is not None:
        print(f"Duplicate listing IDs: {duplicate_ids:,}")

    print(f"Column profile written to: {output_path.resolve()}")

    print(
        f"Parsed non-missing prices: "
        f"{price_numeric.notna().sum():,}"
    )
    print(
        "Price-missingness summary written to: "
        f"{price_output_path.resolve()}"
    )
    print(
        "Price-representation checks written to: "
        f"{price_checks_path.resolve()}"
    )


if __name__ == "__main__":
    main()