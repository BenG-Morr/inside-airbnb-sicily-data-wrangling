"""Basic profiling of the Inside Airbnb Sicily listings dataset.

This script loads the raw listings dataset without modifying it and creates
a column-level profile containing data types, missing-value counts and
uniqueness information. It also reports basic row-level uniqueness checks.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# Airbnb uses these textual labels for half bathrooms without including
# an explicit numeric value. Their interpretation as 0.5 is validated
# against rows where both textual and numeric bathroom data are available.
HALF_BATH_LABELS = {
    "half-bath",
    "private half-bath",
    "shared half-bath",
}


# These attributes describe the host rather than an individual listing.
# Because they are repeated across listing rows, their consistency must
# be checked before they can safely be moved into a separate hosts table.
HOST_FIELDS_FOR_CONSISTENCY = [
    "host_profile_id",
    "host_profile_url",
    "host_name",
    "host_location",
    "host_about",
    "host_is_superhost",
    "host_picture_url",
    "host_listings_count",
    "host_has_profile_pic",
    "host_identity_verified",
    "hosts_time_as_user_years",
    "hosts_time_as_user_months",
    "hosts_time_as_host_years",
    "hosts_time_as_host_months",
]


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


def parse_bathroom_count(value: object) -> float | None:
    """Extract a numeric bathroom count from `bathrooms_text`.

    Most values begin with a numeric count, such as `1 bath`,
    `2 shared baths` or `4.5 baths`. Airbnb also uses textual
    half-bath labels without a leading number; these are represented
    as 0.5 so that they can be compared with the numeric field.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    # These labels contain no numeric prefix, so they require explicit
    # handling rather than the regular-expression parser below.
    if text in HALF_BATH_LABELS:
        return 0.5

    # Standard Airbnb bathroom descriptions start with the bathroom
    # count. The expression also accepts decimal counts such as 1.5.
    match = re.match(r"^(\d+(?:\.\d+)?)\b", text)

    if match is None:
        return None

    return float(match.group(1))


def parse_amenities_cell(
    value: object,
) -> tuple[str, list[object] | None]:
    """Parse one amenities cell and classify its source structure.

    The status is returned separately from the parsed value so that
    genuinely missing data, malformed JSON and valid JSON with the wrong
    structure are not collapsed into the same category.
    """
    if pd.isna(value):
        return "missing", None

    try:
        parsed_value = json.loads(str(value))
    except json.JSONDecodeError:
        return "parse_error", None

    # A successfully parsed JSON value is not necessarily a list. Since
    # the intended structure is one list of amenities per listing, other
    # JSON structures are recorded separately.
    if not isinstance(parsed_value, list):
        return "non_list", None

    return "list", parsed_value


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


def build_bathroom_checks(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare numeric and textual bathroom representations.

    The textual field is used only as an additional source of information.
    Existing numeric bathroom values are not overwritten. Conflicting
    representations are reported separately so that they can later be
    investigated rather than resolved automatically.
    """
    required_columns = {
        "id",
        "bathrooms",
        "bathrooms_text",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing_names}")

    parsed_text = data["bathrooms_text"].map(
        parse_bathroom_count
    )

    numeric_missing = data["bathrooms"].isna()
    text_missing = data["bathrooms_text"].isna()

    # Distinguish absent bathroom text from text that exists but cannot
    # be interpreted by the parser. This makes it possible to determine
    # whether unrecoverable numeric values result from missing source
    # information or from limitations in the parsing logic.
    text_present_but_unparseable = (
            data["bathrooms_text"].notna()
            & parsed_text.isna()
    )
    numeric_and_text_missing = (
            numeric_missing & text_missing
    )

    # A missing numeric value is considered recoverable only when the
    # textual bathroom description can be parsed successfully.
    recoverable_from_text = (
        numeric_missing & parsed_text.notna()
    )

    # Agreement can only be assessed where both representations provide
    # a numeric bathroom count.
    comparable_values = (
        data["bathrooms"].notna()
        & parsed_text.notna()
    )

    # Validate the interpretation of Airbnb's textual half-bath labels
    # against rows where an original numeric bathroom count is available.
    normalised_bathroom_text = (
        data["bathrooms_text"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    half_bath_rows = normalised_bathroom_text.isin(
        HALF_BATH_LABELS
    )
    comparable_half_bath_rows = (
            half_bath_rows & data["bathrooms"].notna()
    )
    half_bath_mismatches = (
        data.loc[comparable_half_bath_rows, "bathrooms"]
        .ne(0.5)
    )

    numeric_difference = (
        data.loc[comparable_values, "bathrooms"]
        - parsed_text[comparable_values]
    ).abs()

    # A small tolerance prevents irrelevant floating-point differences
    # from being classified as genuine discrepancies.
    bathroom_tolerance = 0.001
    conflicts = numeric_difference > bathroom_tolerance

    checks = [
        {
            "check": "missing_numeric_bathrooms",
            "count": int(numeric_missing.sum()),
        },
        {
            "check": "missing_bathrooms_text",
            "count": int(text_missing.sum()),
        },
        {
            "check": "recoverable_missing_numeric_bathrooms",
            "count": int(recoverable_from_text.sum()),
        },
        {
            "check": "missing_numeric_and_text",
            "count": int(numeric_and_text_missing.sum()),
        },
        {
            "check": "present_bathroom_text_parse_failures",
            "count": int(text_present_but_unparseable.sum()),
        },
        {
            "check": "comparable_half_bath_labels",
            "count": int(comparable_half_bath_rows.sum()),
        },
        {
            "check": "half_bath_label_numeric_mismatches",
            "count": int(half_bath_mismatches.sum()),
        },
        {
            "check": "comparable_bathroom_values",
            "count": int(comparable_values.sum()),
        },
        {
            "check": "matching_bathroom_values",
            "count": int((~conflicts).sum()),
        },
        {
            "check": "conflicting_bathroom_values",
            "count": int(conflicts.sum()),
        },
    ]

    # Keep the original and parsed representations side by side so that
    # every disagreement remains transparent and manually inspectable.
    conflict_indices = numeric_difference[conflicts].index
    discrepancies = data.loc[
        conflict_indices,
        ["id", "bathrooms", "bathrooms_text"],
    ].copy()

    discrepancies["parsed_bathrooms_text"] = (
        parsed_text.loc[conflict_indices]
    )

    return pd.DataFrame(checks), discrepancies


def build_host_consistency_checks(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether host attributes are consistent within each host ID.

    Host-level attributes are repeated across listing rows. For a later
    normalisation into a separate hosts table, each host should have at
    most one distinct non-missing value per selected attribute.
    """
    if "host_id" not in data.columns:
        raise KeyError("Expected column 'host_id' was not found.")

    results = []

    for field in HOST_FIELDS_FOR_CONSISTENCY:
        if field not in data.columns:
            continue

        # Missing values are deliberately excluded from the distinct-value
        # count. A host having one observed value and additional missing
        # values is incomplete, but it is not internally contradictory.
        distinct_values_per_host = (
            data.groupby("host_id")[field]
            .nunique(dropna=True)
        )

        # Recording coverage as well as conflicts prevents a field with
        # little observed data from appearing misleadingly consistent.
        hosts_with_non_missing_value = (
            distinct_values_per_host > 0
        ).sum()

        hosts_with_conflicts = (
            distinct_values_per_host > 1
        ).sum()

        results.append(
            {
                "field": field,
                "hosts_with_non_missing_value": int(
                    hosts_with_non_missing_value
                ),
                "hosts_with_multiple_non_missing_values": int(
                    hosts_with_conflicts
                ),
                "max_distinct_values_per_host": int(
                    distinct_values_per_host.max()
                ),
            }
        )

    return pd.DataFrame(results)


def build_amenities_checks(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Assess the structure of the listing amenities field.

    Amenities are stored as multiple values within a single source cell.
    Before considering a later listing-amenity table, this function checks
    whether those values can be parsed consistently as JSON lists.
    """
    if "amenities" not in data.columns:
        raise KeyError("Expected column 'amenities' was not found.")

    parsed_cells = data["amenities"].map(parse_amenities_cell)

    # Keep parsing status separate from the parsed list so that structural
    # problems can be counted without altering the source field.
    parse_status = parsed_cells.map(
        lambda result: result[0]
    )
    amenity_lists = parsed_cells.map(
        lambda result: result[1]
    )

    valid_list_mask = parse_status.eq("list")
    valid_lists = amenity_lists[valid_list_mask]

    # List lengths describe the number of amenity values represented in
    # each source cell. They do not yet create or modify any observations.
    amenities_per_listing = valid_lists.map(len)

    # The later normalisation step assumes individual amenities are strings.
    # Check that assumption explicitly rather than relying on the JSON
    # parser alone.
    non_string_items = valid_lists.map(
        lambda values: sum(
            not isinstance(item, str)
            for item in values
        )
    )

    # A repeated amenity within the same listing would create duplicate
    # listing-amenity pairs after normalisation. Compare each list length
    # with the number of distinct values before creating that table.
    duplicate_items_per_listing = valid_lists.map(
        lambda values: len(values) - len(set(values))
    )
    listings_with_duplicate_amenities = (
            duplicate_items_per_listing > 0
    )

    checks = [
        {
            "check": "missing_amenities",
            "value": int(
                parse_status.eq("missing").sum()
            ),
        },
        {
            "check": "valid_json_lists",
            "value": int(valid_list_mask.sum()),
        },
        {
            "check": "json_parse_errors",
            "value": int(
                parse_status.eq("parse_error").sum()
            ),
        },
        {
            "check": "valid_json_non_lists",
            "value": int(
                parse_status.eq("non_list").sum()
            ),
        },
        {
            "check": "non_string_amenity_items",
            "value": int(non_string_items.sum()),
        },
        {
            "check": "listings_with_duplicate_amenities",
            "value": int(
                listings_with_duplicate_amenities.sum()
            ),
        },
        {
            "check": "duplicate_amenity_items",
            "value": int(
                duplicate_items_per_listing.sum()
            ),
        },
        {
            "check": "median_amenities_per_listing",
            "value": float(amenities_per_listing.median()),
        },
        {
            "check": "maximum_amenities_per_listing",
            "value": int(amenities_per_listing.max()),
        },
    ]

    checks_frame = pd.DataFrame(checks)

    # Store whole-number results without a decimal suffix while preserving
    # non-integer statistics, such as a possible median of 31.5.
    checks_frame["value"] = checks_frame["value"].map(
        lambda value: (
            str(int(value))
            if float(value).is_integer()
            else str(value)
        )
    )

    return checks_frame


def build_review_missingness_checks(
        data: pd.DataFrame,
) -> pd.DataFrame:
    """Assess whether missing review ratings are structurally meaningful.

    Review ratings are only expected once a listing has received at least
    one review. The checks therefore compare missing rating values with
    `number_of_reviews` instead of treating all missing ratings as data
    quality defects.
    """
    required_columns = {
        "review_scores_rating",
        "number_of_reviews",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns: {missing_names}")

    rating_missing = data["review_scores_rating"].isna()
    zero_reviews = data["number_of_reviews"].eq(0)

    # Missing ratings may be semantically meaningful when a listing has
    # never received a review. Such values should not automatically be
    # treated as errors or candidates for imputation.
    missing_rating_with_zero_reviews = (
            rating_missing & zero_reviews
    )

    # A missing rating despite existing reviews would represent a
    # different form of missingness and would require investigation.
    missing_rating_with_reviews = (
            rating_missing & ~zero_reviews
    )

    # A populated rating for a listing with zero reviews would indicate
    # an internal inconsistency between the two review-related fields.
    rating_present_with_zero_reviews = (
            data["review_scores_rating"].notna()
            & zero_reviews
    )

    checks = [
        {
            "check": "missing_review_ratings",
            "count": int(rating_missing.sum()),
        },
        {
            "check": "listings_with_zero_reviews",
            "count": int(zero_reviews.sum()),
        },
        {
            "check": "missing_rating_with_zero_reviews",
            "count": int(
                missing_rating_with_zero_reviews.sum()
            ),
        },
        {
            "check": "missing_rating_with_reviews",
            "count": int(
                missing_rating_with_reviews.sum()
            ),
        },
        {
            "check": "rating_present_with_zero_reviews",
            "count": int(
                rating_present_with_zero_reviews.sum()
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


def write_review_missingness_checks(
    checks: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write review-score missingness checks to a CSV file."""
    output_path = output_dir / "review_missingness_checks.csv"
    checks.to_csv(output_path, index=False)

    return output_path


def write_bathroom_checks(
    checks: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write the summary of bathroom-field diagnostics."""
    output_path = output_dir / "bathroom_checks.csv"
    checks.to_csv(output_path, index=False)

    return output_path


def write_bathroom_discrepancies(
    discrepancies: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write conflicting bathroom representations for inspection."""
    output_path = output_dir / "bathroom_discrepancies.csv"
    discrepancies.to_csv(output_path, index=False)

    return output_path


def write_host_consistency_checks(
    checks: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write host-field consistency diagnostics to a CSV file."""
    output_path = output_dir / "host_field_consistency.csv"
    checks.to_csv(output_path, index=False)

    return output_path


def write_amenities_checks(
    checks: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write amenities-structure diagnostics to a CSV file."""
    output_path = output_dir / "amenities_checks.csv"
    checks.to_csv(output_path, index=False)

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

    # Review-score missingness is assessed separately because a missing
    # rating may be semantically expected for listings without reviews.
    review_checks = build_review_missingness_checks(data)

    # Bathroom completeness and consistency require comparison of the
    # numeric field with its textual counterpart.
    bathroom_checks, bathroom_discrepancies = (
        build_bathroom_checks(data)
    )

    # Repeated host attributes are checked before any later attempt to
    # normalise them into a separate host-level table.
    host_checks = build_host_consistency_checks(data)

    # Amenities contain multiple values within each source cell. Their
    # structure is validated before considering later normalisation.
    amenities_checks = build_amenities_checks(data)

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

    review_checks_path = write_review_missingness_checks(
        review_checks,
        args.output_dir,
    )

    bathroom_checks_path = write_bathroom_checks(
        bathroom_checks,
        args.output_dir,
    )
    bathroom_discrepancies_path = (
        write_bathroom_discrepancies(
            bathroom_discrepancies,
            args.output_dir,
        )
    )

    host_checks_path = write_host_consistency_checks(
        host_checks,
        args.output_dir,
    )

    amenities_checks_path = write_amenities_checks(
        amenities_checks,
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
    print(
        "Review-missingness checks written to: "
        f"{review_checks_path.resolve()}"
    )
    print(
        "Bathroom checks written to: "
        f"{bathroom_checks_path.resolve()}"
    )
    print(
        "Bathroom discrepancies written to: "
        f"{bathroom_discrepancies_path.resolve()}"
    )
    print(
        "Host consistency checks written to: "
        f"{host_checks_path.resolve()}"
    )
    print(
        "Amenities checks written to: "
        f"{amenities_checks_path.resolve()}"
    )


if __name__ == "__main__":
    main()
