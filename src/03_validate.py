"""Validate the processed Inside Airbnb Sicily datasets.

This first validation stage checks the basic structure of the refined
listing, host and listing-amenity tables. More detailed relational and
transformation-specific checks are added in subsequent development stages.
"""

import argparse
from pathlib import Path

import pandas as pd


# The assignment uses one fixed Sicily dataset snapshot. These expected
# counts provide a simple safeguard against accidental row loss or
# duplication during the preceding cleaning and normalisation stages.
EXPECTED_LISTING_ROWS = 56_873
EXPECTED_HOST_ROWS = 29_047

DATE_FIELDS = [
    "last_scraped",
    "calendar_last_scraped",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "first_review",
    "last_review",
]

LISTING_BOOLEAN_FIELDS = [
    "has_availability",
]

HOST_BOOLEAN_FIELDS = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
]

HOST_FIELDS_NORMALISED = [
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
    """Parse command-line paths for the three processed datasets."""
    parser = argparse.ArgumentParser(
        description="Validate the processed Airbnb Sicily datasets."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Path to the original raw listings CSV file.",
    )
    parser.add_argument(
        "--listings",
        type=Path,
        required=True,
        help="Path to the cleaned listings CSV file.",
    )
    parser.add_argument(
        "--hosts",
        type=Path,
        required=True,
        help="Path to the normalised hosts CSV file.",
    )
    parser.add_argument(
        "--amenities",
        type=Path,
        required=True,
        help="Path to the normalised listing-amenities CSV file.",
    )

    return parser.parse_args()


def load_dataset(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Load one processed dataset and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def validate_required_columns(
    listings: pd.DataFrame,
    hosts: pd.DataFrame,
    listing_amenities: pd.DataFrame,
) -> None:
    """Check that each processed table contains its required key columns."""
    required_columns = {
        "listings": (
            listings,
            {"id", "host_id"},
        ),
        "hosts": (
            hosts,
            {"host_id"},
        ),
        "listing_amenities": (
            listing_amenities,
            {"listing_id", "amenity"},
        ),
    }

    for dataset_name, (
        data,
        expected_columns,
    ) in required_columns.items():
        missing_columns = expected_columns.difference(
            data.columns
        )

        if missing_columns:
            missing_names = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                f"{dataset_name} is missing required columns: "
                f"{missing_names}"
            )


def validate_table_sizes(
    listings: pd.DataFrame,
    hosts: pd.DataFrame,
) -> None:
    """Check snapshot-level listing and host row counts."""
    if len(listings) != EXPECTED_LISTING_ROWS:
        raise ValueError(
            "Unexpected number of listing rows: "
            f"{len(listings):,}"
        )

    if len(hosts) != EXPECTED_HOST_ROWS:
        raise ValueError(
            "Unexpected number of host rows: "
            f"{len(hosts):,}"
        )


def validate_primary_keys(
    listings: pd.DataFrame,
    hosts: pd.DataFrame,
) -> None:
    """Check uniqueness and completeness of listing and host identifiers."""
    if listings["id"].isna().any():
        raise ValueError(
            "The listings table contains missing listing IDs."
        )

    if listings["id"].duplicated().any():
        duplicate_count = int(
            listings["id"].duplicated().sum()
        )
        raise ValueError(
            f"Found {duplicate_count} duplicate listing IDs."
        )

    if hosts["host_id"].isna().any():
        raise ValueError(
            "The hosts table contains missing host IDs."
        )

    if hosts["host_id"].duplicated().any():
        duplicate_count = int(
            hosts["host_id"].duplicated().sum()
        )
        raise ValueError(
            f"Found {duplicate_count} duplicate host IDs."
        )


def validate_referential_integrity(
    listings: pd.DataFrame,
    hosts: pd.DataFrame,
    listing_amenities: pd.DataFrame,
) -> None:
    """Check foreign-key relationships between the processed tables.

    Each listing must reference an existing host, and each listing-amenity
    relationship must reference an existing listing. Missing references
    indicate that structural normalisation has orphaned records.
    """
    # `host_id` acts as a foreign key from listings to the normalised
    # host table. Every listing should resolve to exactly one host record.
    missing_host_references = (
        ~listings["host_id"].isin(hosts["host_id"])
    )

    if missing_host_references.any():
        missing_count = int(
            missing_host_references.sum()
        )
        raise ValueError(
            f"{missing_count} listings reference missing hosts."
        )

    # `listing_id` links each amenity relationship back to its parent
    # listing. No relationship row should refer to a listing that was
    # removed or lost during cleaning.
    missing_listing_references = (
        ~listing_amenities["listing_id"].isin(
            listings["id"]
        )
    )

    if missing_listing_references.any():
        missing_count = int(
            missing_listing_references.sum()
        )
        raise ValueError(
            f"{missing_count} amenity rows reference missing listings."
        )


def validate_cleaning_results(
    raw: pd.DataFrame,
    listings: pd.DataFrame,
) -> None:
    """Validate selected cleaning outcomes against the raw dataset."""
    if not raw["id"].reset_index(drop=True).equals(
        listings["id"].reset_index(drop=True)
    ):
        raise ValueError(
            "Listing IDs or row order changed during cleaning."
        )

    required_processed_columns = {
        "price_eur",
        "bathrooms_clean",
        "bathrooms_derived_from_text",
        "bathroom_conflict_flag",
    }
    missing_columns = required_processed_columns.difference(
        listings.columns
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise ValueError(
            "Processed listings are missing cleaning fields: "
            f"{missing_names}"
        )

    # Convert the raw display representation independently and confirm
    # that cleaning neither lost observed prices nor changed their values.
    raw_price_numeric = pd.to_numeric(
        raw["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    if raw_price_numeric.notna().sum() != listings["price_eur"].notna().sum():
        raise ValueError(
            "The number of observed numeric prices changed during cleaning."
        )

    price_comparison = pd.DataFrame(
        {
            "raw_price": raw_price_numeric,
            "clean_price": listings["price_eur"],
        }
    ).dropna()

    price_difference = (
        price_comparison["raw_price"]
        - price_comparison["clean_price"]
    ).abs()

    if price_difference.gt(0.001).any():
        raise ValueError(
            "Processed price_eur values differ from the raw prices."
        )

    # These snapshot-specific counts were established during profiling and
    # provide regression checks for the documented bathroom treatment.
    derived_bathrooms = int(
        listings["bathrooms_derived_from_text"].sum()
    )
    remaining_missing = int(
        listings["bathrooms_clean"].isna().sum()
    )
    bathroom_conflicts = int(
        listings["bathroom_conflict_flag"].sum()
    )

    if derived_bathrooms != 8_792:
        raise ValueError(
            "Unexpected number of bathrooms derived from text: "
            f"{derived_bathrooms:,}"
        )

    if remaining_missing != 55:
        raise ValueError(
            "Unexpected number of missing cleaned bathroom values: "
            f"{remaining_missing:,}"
        )

    if bathroom_conflicts != 6:
        raise ValueError(
            "Unexpected number of bathroom conflicts: "
            f"{bathroom_conflicts:,}"
        )

    # Structural normalisation should remove repeated or nested attributes
    # from the listing-level table while preserving their separate tables.
    forbidden_listing_columns = {
        "amenities",
        *HOST_FIELDS_NORMALISED,
    }
    remaining_columns = forbidden_listing_columns.intersection(
        listings.columns
    )

    if remaining_columns:
        remaining_names = ", ".join(
            sorted(remaining_columns)
        )
        raise ValueError(
            "Normalised fields remain in the listings table: "
            f"{remaining_names}"
        )


def validate_date_fields(
    raw: pd.DataFrame,
    listings: pd.DataFrame,
) -> None:
    """Check that selected dates remain valid and unchanged."""
    for field in DATE_FIELDS:
        if field not in raw.columns or field not in listings.columns:
            raise KeyError(
                f"Required date field is missing: {field}"
            )

        raw_parsed = pd.to_datetime(
            raw[field],
            format="%Y-%m-%d",
            errors="coerce",
        )
        processed_parsed = pd.to_datetime(
            listings[field],
            format="%Y-%m-%d",
            errors="coerce",
        )

        # A non-missing source value that cannot be parsed would make the
        # raw field unsuitable as a reliable validation baseline.
        raw_parse_failures = (
            raw[field].notna()
            & raw_parsed.isna()
        )

        if raw_parse_failures.any():
            failure_count = int(
                raw_parse_failures.sum()
            )
            raise ValueError(
                f"{field} contains {failure_count} raw parse failures."
            )

        # Cleaning must not turn an observed date into a missing value.
        processed_parse_failures = (
            listings[field].notna()
            & processed_parsed.isna()
        )

        if processed_parse_failures.any():
            failure_count = int(
                processed_parse_failures.sum()
            )
            raise ValueError(
                f"{field} contains {failure_count} processed "
                "parse failures."
            )

        if not raw[field].isna().equals(
            listings[field].isna()
        ):
            raise ValueError(
                f"Missingness changed for date field {field}."
            )

        # Because listing row order is validated separately, the parsed
        # dates can be compared position by position with the source.
        if not raw_parsed.equals(processed_parsed):
            raise ValueError(
                f"Date values changed during cleaning for {field}."
            )


def validate_review_missingness(
    raw: pd.DataFrame,
    listings: pd.DataFrame,
) -> None:
    """Validate the structural missingness of review-score ratings."""
    required_columns = {
        "review_scores_rating",
        "number_of_reviews",
    }

    for field in required_columns:
        if field not in raw.columns or field not in listings.columns:
            raise KeyError(
                f"Required review field is missing: {field}"
            )

    raw_missing = raw["review_scores_rating"].isna()
    processed_missing = listings[
        "review_scores_rating"
    ].isna()

    # Review-score missingness was intentionally retained rather than
    # imputed, so its position and frequency must remain unchanged.
    if not raw_missing.equals(processed_missing):
        raise ValueError(
            "Review-score missingness changed during cleaning."
        )

    zero_reviews = listings["number_of_reviews"].eq(0)

    # Profiling established that missing ratings correspond exactly to
    # listings with no reviews. This is therefore meaningful structural
    # missingness rather than evidence for arbitrary imputation.
    if not processed_missing.equals(zero_reviews):
        raise ValueError(
            "Missing review ratings no longer correspond exactly "
            "to listings with zero reviews."
        )

    missing_count = int(
        processed_missing.sum()
    )

    if missing_count != 16_501:
        raise ValueError(
            "Unexpected number of missing review ratings: "
            f"{missing_count:,}"
        )


def canonicalise_raw_boolean(
    series: pd.Series,
    field: str,
) -> pd.Series:
    """Map raw t/f values to a nullable Boolean representation."""
    unexpected = (
        series.notna()
        & ~series.isin(["t", "f"])
    )

    if unexpected.any():
        unexpected_values = sorted(
            series.loc[unexpected].astype(str).unique()
        )
        raise ValueError(
            f"Unexpected raw Boolean values in {field}: "
            f"{unexpected_values}"
        )

    return series.map(
        {
            "t": True,
            "f": False,
        }
    ).astype("boolean")


def canonicalise_processed_boolean(
    series: pd.Series,
    field: str,
) -> pd.Series:
    """Map processed Boolean values to one comparable representation."""
    mapped = series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
        }
    )

    unexpected = (
        series.notna()
        & mapped.isna()
    )

    if unexpected.any():
        unexpected_values = sorted(
            series.loc[unexpected].astype(str).unique()
        )
        raise ValueError(
            f"Unexpected processed Boolean values in {field}: "
            f"{unexpected_values}"
        )

    return mapped.astype("boolean")


def validate_boolean_fields(
    raw: pd.DataFrame,
    listings: pd.DataFrame,
    hosts: pd.DataFrame,
) -> None:
    """Validate Boolean harmonisation in the processed tables."""
    for field in LISTING_BOOLEAN_FIELDS:
        if field not in raw.columns or field not in listings.columns:
            raise KeyError(
                f"Required listing Boolean field is missing: {field}"
            )

        expected = canonicalise_raw_boolean(
            raw[field],
            field,
        )
        actual = canonicalise_processed_boolean(
            listings[field],
            field,
        )

        # Listing row order is validated elsewhere, so Boolean values can
        # be compared directly with their corresponding source rows.
        if not expected.reset_index(drop=True).equals(
            actual.reset_index(drop=True)
        ):
            raise ValueError(
                f"Boolean values changed during cleaning for {field}."
            )

    for field in HOST_BOOLEAN_FIELDS:
        if field not in raw.columns or field not in hosts.columns:
            raise KeyError(
                f"Required host Boolean field is missing: {field}"
            )

        raw_values = canonicalise_raw_boolean(
            raw[field],
            field,
        )

        raw_host_values = pd.DataFrame(
            {
                "host_id": raw["host_id"],
                field: raw_values,
            }
        )

        # Host attributes occurred repeatedly in the raw listing table.
        # Verify consistency before deriving one expected value per host.
        distinct_values = (
            raw_host_values.groupby("host_id")[field]
            .nunique(dropna=True)
        )

        if distinct_values.gt(1).any():
            conflict_count = int(
                distinct_values.gt(1).sum()
            )
            raise ValueError(
                f"{field} has conflicting values for "
                f"{conflict_count} hosts."
            )

        expected_by_host = (
            raw_host_values.groupby("host_id")[field]
            .first()
            .sort_index()
            .astype("boolean")
        )

        processed_values = canonicalise_processed_boolean(
            hosts[field],
            field,
        )
        actual_by_host = pd.Series(
            processed_values.array,
            index=hosts["host_id"],
            name=field,
        ).sort_index()

        if not expected_by_host.equals(actual_by_host):
            raise ValueError(
                "Normalised host Boolean values differ from "
                f"the raw data for {field}."
            )


def main() -> None:
    """Run the basic processed-data validation checks."""
    args = parse_arguments()

    raw = load_dataset(
        args.raw,
        "Raw listings",
    )
    listings = load_dataset(
        args.listings,
        "Listings",
    )
    hosts = load_dataset(
        args.hosts,
        "Hosts",
    )
    listing_amenities = load_dataset(
        args.amenities,
        "Listing amenities",
    )

    validate_required_columns(
        listings,
        hosts,
        listing_amenities,
    )
    validate_table_sizes(
        listings,
        hosts,
    )
    validate_primary_keys(
        listings,
        hosts,
    )
    validate_referential_integrity(
        listings,
        hosts,
        listing_amenities,
    )
    validate_cleaning_results(
        raw,
        listings,
    )
    validate_date_fields(
        raw,
        listings,
    )
    validate_review_missingness(
        raw,
        listings,
    )
    validate_boolean_fields(
        raw,
        listings,
        hosts,
    )

    print("Processed-data validation passed.")
    print(f"Listings: {len(listings):,} rows")
    print(f"Hosts: {len(hosts):,} rows")
    print(
        "Listing-amenity relationships: "
        f"{len(listing_amenities):,} rows"
    )
    print(
        f"Date fields validated: {len(DATE_FIELDS)}"
    )
    print(
        "Structurally missing review ratings: "
        f"{listings['review_scores_rating'].isna().sum():,}"
    )
    print(
        "Boolean fields validated: "
        f"{len(LISTING_BOOLEAN_FIELDS) + len(HOST_BOOLEAN_FIELDS)}"
    )


if __name__ == "__main__":
    main()
