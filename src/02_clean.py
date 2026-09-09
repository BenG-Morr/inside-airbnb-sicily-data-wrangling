"""Clean selected fields in the Inside Airbnb Sicily listings dataset.

This first cleaning stage preserves the raw source columns and derives an
explicit numeric EUR price field. The transformation is validated against
the profiling findings before the refined dataset is written to disk.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(
        description="Clean selected fields in the Airbnb listings dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw listings CSV or CSV.GZ file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the cleaned listings CSV file.",
    )

    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the raw listings dataset without modifying the source file."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    return pd.read_csv(input_path, low_memory=False)


def derive_price_eur(data: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric EUR price while preserving the raw price field.

    The raw `price` values contain formatting characters such as a dollar
    sign and thousands separators, although the available quote metadata
    identifies the currency as EUR. The source field is therefore retained
    unchanged and a separate numeric `price_eur` field is derived.
    """
    if "price" not in data.columns:
        raise KeyError("Expected column 'price' was not found.")

    cleaned = data.copy()

    price_text = (
        cleaned["price"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    cleaned["price_eur"] = pd.to_numeric(
        price_text,
        errors="coerce",
    )

    return cleaned


def validate_price_eur(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> None:
    """Validate that deriving `price_eur` does not lose observed prices.

    Any non-missing raw price that becomes missing after numeric conversion
    is treated as a failed transformation. When the numeric quote field is
    available, its value is also compared with the derived EUR price.
    """
    required_columns = {
        "price",
        "price_quote_price_per_night",
    }
    missing_columns = required_columns.difference(original.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required columns: {missing_names}"
        )

    raw_price_present = original["price"].notna()
    conversion_failures = (
        raw_price_present
        & cleaned["price_eur"].isna()
    )

    if conversion_failures.any():
        failure_count = int(conversion_failures.sum())
        raise ValueError(
            f"Price conversion failed for {failure_count} rows."
        )

    comparable_prices = (
        cleaned["price_eur"].notna()
        & original["price_quote_price_per_night"].notna()
    )

    price_difference = (
        cleaned.loc[comparable_prices, "price_eur"]
        - original.loc[
            comparable_prices,
            "price_quote_price_per_night",
        ]
    ).abs()

    # Floating-point calculations can contain insignificant representation
    # differences. A small tolerance prevents these from being treated as
    # genuine price mismatches.
    price_tolerance = 0.001

    mismatches = price_difference > price_tolerance

    if mismatches.any():
        mismatch_count = int(mismatches.sum())
        raise ValueError(
            f"Price validation found {mismatch_count} mismatches."
        )


def write_cleaned_dataset(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the cleaned listings dataset to the requested local path."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data.to_csv(output_path, index=False)


def main() -> None:
    """Run the first stage of the data-cleaning pipeline."""
    args = parse_arguments()

    raw_data = load_dataset(args.input)

    # The transformation returns a copy so that the DataFrame representing
    # the raw source remains unchanged throughout the cleaning workflow.
    cleaned_data = derive_price_eur(raw_data)

    # Validate the derived representation before any processed data are
    # written to disk.
    validate_price_eur(
        raw_data,
        cleaned_data,
    )

    write_cleaned_dataset(
        cleaned_data,
        args.output,
    )

    print("Price cleaning complete.")
    print(f"Rows preserved: {len(cleaned_data):,}")
    print(
        "Non-missing derived prices: "
        f"{cleaned_data['price_eur'].notna().sum():,}"
    )
    print(f"Output written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
