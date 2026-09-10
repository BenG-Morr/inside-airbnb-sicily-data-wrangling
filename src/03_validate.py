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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line paths for the three processed datasets."""
    parser = argparse.ArgumentParser(
        description="Validate the processed Airbnb Sicily datasets."
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


def main() -> None:
    """Run the basic processed-data validation checks."""
    args = parse_arguments()

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

    print("Basic processed-data validation passed.")
    print(f"Listings: {len(listings):,} rows")
    print(f"Hosts: {len(hosts):,} rows")
    print(
        "Listing-amenity relationships: "
        f"{len(listing_amenities):,} rows"
    )


if __name__ == "__main__":
    main()
