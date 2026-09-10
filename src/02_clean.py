"""Clean and structurally tidy the Inside Airbnb Sicily listings dataset.

The pipeline harmonises selected representations, derives validated fields,
normalises host and amenity data into separate tables, removes fields that
are fully empty in the analysed snapshot, and writes processed outputs while
preserving the raw source file.
"""

import argparse
import json
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


# These attributes describe the host rather than an individual listing.
# Profiling showed that each host has at most one distinct non-missing
# value for every selected field, supporting a separate host-level table.
HOST_FIELDS_TO_NORMALISE = [
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


# These fields are entirely missing in the analysed Sicily snapshot and
# therefore provide no information for the defined analytical scope.
# They are removed only after their complete missingness is revalidated.
EMPTY_FIELDS_TO_DROP = [
    "neighborhood_overview",
    "host_since",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_thumbnail_url",
    "host_neighbourhood",
    "host_total_listings_count",
    "host_verifications",
    "neighbourhood",
    "neighbourhood_group_cleansed",
    "calendar_updated",
    "instant_bookable",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Clean and structurally tidy the Airbnb Sicily dataset."
        )
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
    parser.add_argument(
        "--hosts-output",
        type=Path,
        required=True,
        help="Path for the normalised hosts CSV file.",
    )
    parser.add_argument(
        "--amenities-output",
        type=Path,
        required=True,
        help="Path for the normalised listing-amenities CSV file.",
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


def normalise_host_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate host-level attributes from listing-level observations.

    Each listing retains `host_id` as a foreign key. The selected host
    attributes are moved into a table containing one row per host.

    Before collapsing repeated host data, the function verifies that no
    host has more than one distinct non-missing value for any selected
    host field.
    """
    required_columns = {
        "host_id",
        *HOST_FIELDS_TO_NORMALISE,
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required host columns: {missing_names}"
        )

    if data["host_id"].isna().any():
        raise ValueError(
            "Host normalisation requires non-missing host IDs."
        )

    # Collapsing repeated host rows is safe only if the observed values
    # for each host-level attribute are internally consistent.
    conflicting_fields = []

    for field in HOST_FIELDS_TO_NORMALISE:
        distinct_values = (
            data.groupby("host_id")[field]
            .nunique(dropna=True)
        )

        if distinct_values.gt(1).any():
            conflicting_fields.append(field)

    if conflicting_fields:
        conflicting_names = ", ".join(conflicting_fields)
        raise ValueError(
            "Conflicting host-level values found in: "
            f"{conflicting_names}"
        )

    host_columns = [
        "host_id",
        *HOST_FIELDS_TO_NORMALISE,
    ]

    # `groupby().first()` selects the first non-missing value for each
    # field. Because consistency was checked above, this cannot combine
    # contradictory observed values for the same host.
    hosts = (
        data[host_columns]
        .groupby("host_id", as_index=False)
        .first()
    )

    # The listing table keeps host_id but no longer repeats the selected
    # host attributes for every listing.
    listings = data.drop(
        columns=HOST_FIELDS_TO_NORMALISE
    ).copy()

    return listings, hosts


def normalise_amenities(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate nested amenities into listing-amenity observations.

    The source stores all amenities for one listing as a JSON list inside
    a single cell. Each amenity is moved to its own row while the listing
    table retains the listing ID needed to link both tables.

    Amenity strings are preserved exactly as supplied by the source. This
    step changes only the structure and does not standardise amenity names.
    """
    required_columns = {
        "id",
        "amenities",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Missing required amenities columns: {missing_names}"
        )

    amenity_records = []

    for listing_id, raw_amenities in data[
        ["id", "amenities"]
    ].itertuples(index=False, name=None):
        if pd.isna(raw_amenities):
            raise ValueError(
                f"Missing amenities for listing {listing_id}."
            )

        try:
            parsed_amenities = json.loads(
                str(raw_amenities)
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid amenities JSON for listing {listing_id}."
            ) from error

        # Profiling indicated that amenities are consistently represented
        # as lists. The cleaning pipeline nevertheless validates this
        # assumption so that future malformed input is not silently used.
        if not isinstance(parsed_amenities, list):
            raise TypeError(
                f"Amenities are not a list for listing {listing_id}."
            )

        for amenity in parsed_amenities:
            if not isinstance(amenity, str):
                raise TypeError(
                    "Non-string amenity found for listing "
                    f"{listing_id}."
                )

            amenity_records.append(
                {
                    "listing_id": listing_id,
                    "amenity": amenity,
                }
            )

    listing_amenities = pd.DataFrame(
        amenity_records,
        columns=["listing_id", "amenity"],
    )

    # The nested source field is no longer needed in the refined listing
    # table because its observations now reside in listing_amenities.
    listings = data.drop(
        columns=["amenities"]
    ).copy()

    return listings, listing_amenities


def remove_fully_empty_fields(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Remove fields verified as entirely missing in this snapshot."""
    missing_columns = set(
        EMPTY_FIELDS_TO_DROP
    ).difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise KeyError(
            "Configured empty fields are missing from the dataset: "
            f"{missing_names}"
        )

    # A configured field must still be entirely empty before exclusion.
    # This prevents a future dataset snapshot from silently losing newly
    # available information merely because the column name is listed here.
    non_empty_fields = [
        field
        for field in EMPTY_FIELDS_TO_DROP
        if data[field].notna().any()
    ]

    if non_empty_fields:
        non_empty_names = ", ".join(
            sorted(non_empty_fields)
        )
        raise ValueError(
            "Fields configured for removal contain observed values: "
            f"{non_empty_names}"
        )

    return data.drop(
        columns=EMPTY_FIELDS_TO_DROP
    ).copy()


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


def validate_host_normalisation(
    original: pd.DataFrame,
    listings: pd.DataFrame,
    hosts: pd.DataFrame,
) -> None:
    """Validate host-table uniqueness and referential integrity."""
    expected_hosts = int(original["host_id"].nunique())

    if len(hosts) != expected_hosts:
        raise ValueError(
            f"Expected {expected_hosts} hosts, found {len(hosts)}."
        )

    if hosts["host_id"].duplicated().any():
        raise ValueError(
            "The normalised hosts table contains duplicate host IDs."
        )

    # Every listing host_id must resolve to exactly one host-table row.
    missing_host_references = (
        ~listings["host_id"].isin(hosts["host_id"])
    )

    if missing_host_references.any():
        missing_count = int(missing_host_references.sum())
        raise ValueError(
            f"{missing_count} listing host IDs have no host record."
        )

    # The normalised fields should no longer be repeated in listings.
    remaining_host_fields = set(
        HOST_FIELDS_TO_NORMALISE
    ).intersection(listings.columns)

    if remaining_host_fields:
        remaining_names = ", ".join(
            sorted(remaining_host_fields)
        )
        raise ValueError(
            "Host fields remain in the listings table: "
            f"{remaining_names}"
        )


def validate_amenities_normalisation(
    original: pd.DataFrame,
    listings: pd.DataFrame,
    listing_amenities: pd.DataFrame,
) -> None:
    """Validate the listing-amenity table and its relationship to listings."""
    if "amenities" in listings.columns:
        raise ValueError(
            "The nested amenities field remains in the listings table."
        )

    # Count the source amenity observations independently. The number of
    # normalised rows must equal the total number of list elements in the
    # original source field.
    expected_amenity_rows = 0

    for raw_amenities in original["amenities"]:
        parsed_amenities = json.loads(
            str(raw_amenities)
        )
        expected_amenity_rows += len(
            parsed_amenities
        )

    if len(listing_amenities) != expected_amenity_rows:
        raise ValueError(
            "Amenity normalisation changed the number of amenity "
            "observations."
        )

    # Profiling found no repeated amenity within the same listing. The
    # normalised table should therefore contain no duplicate
    # listing-amenity pairs.
    duplicate_pairs = listing_amenities.duplicated(
        subset=["listing_id", "amenity"]
    )

    if duplicate_pairs.any():
        duplicate_count = int(duplicate_pairs.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate listing-amenity pairs."
        )

    # Every foreign key in the amenities table must reference an existing
    # listing in the refined listings table.
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
            f"{missing_count} amenities reference missing listings."
        )

    # Listings with an empty source amenities list legitimately produce no
    # rows in the normalised relationship table. Verify that every listing
    # omitted from listing_amenities is explained by such an empty list.
    source_amenity_counts = original["amenities"].map(
        lambda value: len(json.loads(str(value)))
    )
    zero_amenity_listing_ids = set(
        original.loc[
            source_amenity_counts.eq(0),
            "id",
        ]
    )

    represented_listing_ids = set(
        listing_amenities["listing_id"].unique()
    )
    omitted_listing_ids = (
            set(listings["id"]) - represented_listing_ids
    )

    if omitted_listing_ids != zero_amenity_listing_ids:
        raise ValueError(
            "Listings omitted from the amenity table do not match "
            "the listings with empty source amenity lists."
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


def write_dataset(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a processed dataset to the requested local path."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data.to_csv(output_path, index=False)


def main() -> None:
    """Run the complete cleaning and structural-tidying pipeline."""
    args = parse_arguments()

    raw_data = load_dataset(args.input)

    # Each transformation returns a new DataFrame so that the object
    # representing the raw source remains unchanged throughout the
    # cleaning workflow.
    cleaned_data = derive_price_eur(raw_data)
    cleaned_data = convert_date_fields(cleaned_data)
    cleaned_data = convert_boolean_fields(cleaned_data)
    cleaned_data = derive_bathroom_fields(cleaned_data)

    # Validate field-level transformations before moving host attributes
    # out of the listing-level table.
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

    cleaned_data, hosts_data = normalise_host_data(
        cleaned_data
    )
    validate_host_normalisation(
        raw_data,
        cleaned_data,
        hosts_data,
    )

    # Amenities represent a many-to-many-style relationship between
    # listings and amenity labels and are therefore moved out of the
    # listing-level table.
    cleaned_data, listing_amenities_data = normalise_amenities(
        cleaned_data
    )
    validate_amenities_normalisation(
        raw_data,
        cleaned_data,
        listing_amenities_data,
    )

    cleaned_data = remove_fully_empty_fields(
        cleaned_data
    )

    write_dataset(
        cleaned_data,
        args.output,
    )
    write_dataset(
        hosts_data,
        args.hosts_output,
    )
    write_dataset(
        listing_amenities_data,
        args.amenities_output,
    )

    print("Cleaning and structural tidying complete.")
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
    print(
        f"Normalised hosts: {len(hosts_data):,}")
    print(
        "Listings columns after structural normalisation: "
        f"{len(cleaned_data.columns)}")
    print(
        f"Fully empty fields removed: {len(EMPTY_FIELDS_TO_DROP)}")
    print(
        f"Hosts columns: {len(hosts_data.columns)}")
    print(
        "Normalised listing-amenity rows: "
        f"{len(listing_amenities_data):,}")
    print(
        "Listing-amenity columns: "
        f"{len(listing_amenities_data.columns)}")
    print(f"Listings written to: {args.output.resolve()}")
    print(f"Hosts written to: {args.hosts_output.resolve()}")
    print(
        "Listing amenities written to: "
        f"{args.amenities_output.resolve()}")


if __name__ == "__main__":
    main()
