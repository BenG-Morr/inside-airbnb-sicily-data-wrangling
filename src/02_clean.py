"""Clean selected fields in the Inside Airbnb Sicily listings dataset.

This first cleaning stage preserves the raw source columns and derives an
explicit numeric EUR price field. The transformation is validated against
the profiling findings before the refined dataset is written to disk.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# These date fields were shown during profiling to contain no malformed
# observed values. They can therefore be converted explicitly rather than
# relying on pandas to infer their meaning later.
DATE_FIELDS_TO_CONVERT = [
    "last_scraped",
    "calendar_last_scraped",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "first_review",
    "last_review",
]


# These fields contain only Airbnb's textual "t" and "f" representations
# when observed. `instant_bookable` is deliberately excluded because it is
# entirely missing in this dataset snapshot.
BOOLEAN_FIELDS_TO_CONVERT = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
]


# Airbnb uses these textual labels for half bathrooms without including
# an explicit numeric count. Profiling confirmed that all comparable
# source rows with these labels have a numeric bathroom value of 0.5.
HALF_BATH_LABELS = {
    "half-bath",
    "private half-bath",
    "shared half-bath",
}


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


def convert_date_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Convert selected date strings to explicit datetime values.

    Profiling showed that all observed values in these fields are
    parseable. Using an explicit date format therefore makes the intended
    representation unambiguous while retaining genuine missing values.
    """
    missing_columns = set(DATE_FIELDS_TO_CONVERT).difference(
        data.columns
    )

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required date columns: {missing_names}"
        )

    cleaned = data.copy()

    for field in DATE_FIELDS_TO_CONVERT:
        # `errors="raise"` is deliberate. A future source snapshot with a
        # malformed observed date should stop the pipeline rather than
        # silently turn that value into missing data.
        cleaned[field] = pd.to_datetime(
            cleaned[field],
            format="%Y-%m-%d",
            errors="raise",
        )

    return cleaned


def convert_boolean_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Convert Airbnb's textual boolean fields to nullable booleans.

    Only "t", "f" and missing values are accepted. Unexpected observed
    values cause the pipeline to fail instead of being silently coerced.
    """
    missing_columns = set(BOOLEAN_FIELDS_TO_CONVERT).difference(
        data.columns
    )

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required boolean columns: {missing_names}"
        )

    cleaned = data.copy()
    boolean_mapping = {
        "t": True,
        "f": False,
    }

    for field in BOOLEAN_FIELDS_TO_CONVERT:
        source_values = cleaned[field]

        unexpected_values = source_values[
            source_values.notna()
            & ~source_values.isin(boolean_mapping)
        ]

        if not unexpected_values.empty:
            unexpected = sorted(
                unexpected_values.astype(str).unique()
            )
            raise ValueError(
                f"Unexpected values in {field}: {unexpected}"
            )

        # Pandas' nullable BooleanDtype retains missing values as a third
        # state rather than forcing them to True or False.
        cleaned[field] = (
            source_values
            .map(boolean_mapping)
            .astype("boolean")
        )

    return cleaned


def parse_bathroom_count(value: object) -> float | None:
    """Extract a numeric bathroom count from `bathrooms_text`.

    Standard Airbnb descriptions begin with a numeric count, for example
    `1 bath`, `2 shared baths` or `1.5 baths`. Textual half-bath labels
    require explicit handling because they contain no numeric prefix.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in HALF_BATH_LABELS:
        return 0.5

    match = re.match(r"^(\d+(?:\.\d+)?)\b", text)

    if match is None:
        return None

    return float(match.group(1))


def derive_bathroom_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Derive a consolidated bathroom value and diagnostic flags.

    Existing numeric bathroom values take precedence. Parsed textual
    values are used only where the numeric source field is missing.
    Conflicting populated representations are flagged rather than
    automatically resolved.
    """
    required_columns = {
        "bathrooms",
        "bathrooms_text",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required bathroom columns: {missing_names}"
        )

    cleaned = data.copy()
    parsed_text = cleaned["bathrooms_text"].map(
        parse_bathroom_count
    )

    numeric_missing = cleaned["bathrooms"].isna()
    parsed_text_present = parsed_text.notna()

    # Only missing numeric values are filled from the textual field.
    # Existing numeric values are never overwritten.
    derived_from_text = (
        numeric_missing & parsed_text_present
    )

    cleaned["bathrooms_clean"] = cleaned["bathrooms"]
    cleaned.loc[
        derived_from_text,
        "bathrooms_clean",
    ] = parsed_text.loc[derived_from_text]

    cleaned["bathrooms_derived_from_text"] = (
        derived_from_text.astype("boolean")
    )

    # Conflicts are assessed only where both representations provide
    # numeric information. A small tolerance avoids treating irrelevant
    # floating-point representation differences as genuine conflicts.
    comparable_values = (
        cleaned["bathrooms"].notna()
        & parsed_text.notna()
    )
    bathroom_tolerance = 0.001

    bathroom_difference = (
        cleaned.loc[comparable_values, "bathrooms"]
        - parsed_text.loc[comparable_values]
    ).abs()

    conflict_flag = pd.Series(
        False,
        index=cleaned.index,
        dtype="boolean",
    )
    conflict_flag.loc[
        bathroom_difference.index
    ] = bathroom_difference.gt(bathroom_tolerance)

    cleaned["bathroom_conflict_flag"] = conflict_flag

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


def validate_date_fields(
        original: pd.DataFrame,
        cleaned: pd.DataFrame,
) -> None:
    """Validate date conversion and preservation of missing values."""
    for field in DATE_FIELDS_TO_CONVERT:
        if not pd.api.types.is_datetime64_any_dtype(cleaned[field]):
            raise TypeError(
                f"Date conversion failed for field: {field}"
            )

        original_missing = int(original[field].isna().sum())
        cleaned_missing = int(cleaned[field].isna().sum())

        if original_missing != cleaned_missing:
            raise ValueError(
                f"Date conversion changed missingness in {field}: "
                f"{original_missing} -> {cleaned_missing}"
            )


def validate_boolean_fields(
        original: pd.DataFrame,
        cleaned: pd.DataFrame,
) -> None:
    """Validate Boolean counts and preservation of missing values."""
    for field in BOOLEAN_FIELDS_TO_CONVERT:
        if str(cleaned[field].dtype) != "boolean":
            raise TypeError(
                f"Boolean conversion failed for field: {field}"
            )

        expected_true = int(original[field].eq("t").sum())
        expected_false = int(original[field].eq("f").sum())
        expected_missing = int(original[field].isna().sum())

        actual_true = int(cleaned[field].eq(True).sum())
        actual_false = int(cleaned[field].eq(False).sum())
        actual_missing = int(cleaned[field].isna().sum())

        if (
            expected_true != actual_true
            or expected_false != actual_false
            or expected_missing != actual_missing
        ):
            raise ValueError(
                f"Boolean validation failed for field: {field}"
            )


def validate_bathroom_fields(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> None:
    """Validate bathroom derivation against the profiling results."""
    original_present = original["bathrooms"].notna()

    # Existing numeric bathroom values must remain unchanged in the
    # consolidated field, including rows with conflicting text.
    changed_existing_values = (
        cleaned.loc[original_present, "bathrooms_clean"]
        .ne(original.loc[original_present, "bathrooms"])
    )

    if changed_existing_values.any():
        raise ValueError(
            "Bathroom derivation changed existing numeric values."
        )

    derived_count = int(
        cleaned["bathrooms_derived_from_text"].sum()
    )
    remaining_missing = int(
        cleaned["bathrooms_clean"].isna().sum()
    )
    conflict_count = int(
        cleaned["bathroom_conflict_flag"].sum()
    )

    # These expected counts were established independently during the
    # profiling stage for this dataset snapshot.
    if derived_count != 8792:
        raise ValueError(
            f"Expected 8792 derived bathrooms, found {derived_count}."
        )

    if remaining_missing != 55:
        raise ValueError(
            f"Expected 55 missing bathrooms, found {remaining_missing}."
        )

    if conflict_count != 6:
        raise ValueError(
            f"Expected 6 bathroom conflicts, found {conflict_count}."
        )


def validate_date_fields(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> None:
    """Validate date conversion and preservation of missing values."""
    for field in DATE_FIELDS_TO_CONVERT:
        if not pd.api.types.is_datetime64_any_dtype(cleaned[field]):
            raise TypeError(
                f"Date conversion failed for field: {field}"
            )

        original_missing = int(original[field].isna().sum())
        cleaned_missing = int(cleaned[field].isna().sum())

        if original_missing != cleaned_missing:
            raise ValueError(
                f"Date conversion changed missingness in {field}: "
                f"{original_missing} -> {cleaned_missing}"
            )


def validate_boolean_fields(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> None:
    """Validate Boolean counts and preservation of missing values."""
    for field in BOOLEAN_FIELDS_TO_CONVERT:
        if str(cleaned[field].dtype) != "boolean":
            raise TypeError(
                f"Boolean conversion failed for field: {field}"
            )

        expected_true = int(original[field].eq("t").sum())
        expected_false = int(original[field].eq("f").sum())
        expected_missing = int(original[field].isna().sum())

        actual_true = int(cleaned[field].eq(True).sum())
        actual_false = int(cleaned[field].eq(False).sum())
        actual_missing = int(cleaned[field].isna().sum())

        if (
            expected_true != actual_true
            or expected_false != actual_false
            or expected_missing != actual_missing
        ):
            raise ValueError(
                f"Boolean validation failed for field: {field}"
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

    # Each transformation returns a new DataFrame so that the object
    # representing the raw source remains unchanged throughout the
    # cleaning workflow.
    cleaned_data = derive_price_eur(raw_data)
    cleaned_data = convert_date_fields(cleaned_data)
    cleaned_data = convert_boolean_fields(cleaned_data)
    cleaned_data = derive_bathroom_fields(cleaned_data)

    # Validate every transformation before processed data are written.
    validate_price_eur(
        raw_data,
        cleaned_data,
    )
    validate_date_fields(
        raw_data,
        cleaned_data,
    )
    validate_boolean_fields(
        raw_data,
        cleaned_data,
    )
    validate_bathroom_fields(
        raw_data,
        cleaned_data,
    )

    write_cleaned_dataset(
        cleaned_data,
        args.output,
    )

    print("Representation harmonisation complete.")
    print(f"Rows preserved: {len(cleaned_data):,}")
    print(
        "Non-missing derived prices: "
        f"{cleaned_data['price_eur'].notna().sum():,}"
    )
    print(
        f"Date fields converted: {len(DATE_FIELDS_TO_CONVERT)}"
    )
    print(
        f"Boolean fields converted: {len(BOOLEAN_FIELDS_TO_CONVERT)}"
    )
    print(
        "Bathroom values derived from text: "
        f"{cleaned_data['bathrooms_derived_from_text'].sum():,}")
    print(
        "Remaining missing bathroom values: "
        f"{cleaned_data['bathrooms_clean'].isna().sum():,}")
    print(
        "Bathroom conflicts flagged: "
        f"{cleaned_data['bathroom_conflict_flag'].sum():,}")
    print(f"Output written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
