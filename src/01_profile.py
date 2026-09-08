"""Basic profiling of the Inside Airbnb Sicily listings dataset.

This script loads the raw listings dataset without modifying it and creates
a column-level profile containing data types, missing-value counts and
uniqueness information. It also reports basic row-level uniqueness checks.
"""

import argparse
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


if __name__ == "__main__":
    main()