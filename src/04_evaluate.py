"""Evaluate the refined Inside Airbnb Sicily dataset.

This first evaluation stage summarises the distribution of the cleaned
numeric price variable. Both conventional and robust statistics are
reported because extreme observations may strongly influence measures
such as the arithmetic mean without necessarily representing errors.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Evaluate the refined Airbnb Sicily dataset."
    )
    parser.add_argument(
        "--listings",
        type=Path,
        required=True,
        help="Path to the cleaned listings CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for evaluation outputs.",
    )

    return parser.parse_args()


def load_listings(path: Path) -> pd.DataFrame:
    """Load the refined listings dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Listings file not found: {path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def build_price_summary(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise the cleaned numeric price distribution."""
    if "price_eur" not in listings.columns:
        raise KeyError(
            "Required field price_eur is missing."
        )

    prices = listings["price_eur"].dropna()

    if prices.empty:
        raise ValueError(
            "No non-missing price_eur values are available."
        )

    quantiles = prices.quantile(
        [
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            0.995,
            0.999,
        ]
    )

    summary = [
        {
            "metric": "count",
            "value": len(prices),
        },
        {
            "metric": "missing",
            "value": listings["price_eur"].isna().sum(),
        },
        {
            "metric": "mean",
            "value": prices.mean(),
        },
        {
            "metric": "median",
            "value": quantiles.loc[0.50],
        },
        {
            "metric": "standard_deviation",
            "value": prices.std(),
        },
        {
            "metric": "minimum",
            "value": prices.min(),
        },
        {
            "metric": "p25",
            "value": quantiles.loc[0.25],
        },
        {
            "metric": "p75",
            "value": quantiles.loc[0.75],
        },
        {
            "metric": "iqr",
            "value": (
                quantiles.loc[0.75]
                - quantiles.loc[0.25]
            ),
        },
        {
            "metric": "p95",
            "value": quantiles.loc[0.95],
        },
        {
            "metric": "p99",
            "value": quantiles.loc[0.99],
        },
        {
            "metric": "p99_5",
            "value": quantiles.loc[0.995],
        },
        {
            "metric": "p99_9",
            "value": quantiles.loc[0.999],
        },
        {
            "metric": "maximum",
            "value": prices.max(),
        },
    ]

    return pd.DataFrame(summary)


def build_price_sensitivity(
    listings: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how upper-tail restrictions affect price statistics.

    The thresholds are derived from the complete observed price
    distribution. Restricted scenarios are used only for sensitivity
    analysis and do not modify the refined dataset.
    """
    prices = listings["price_eur"].dropna()

    if prices.empty:
        raise ValueError(
            "No non-missing price_eur values are available."
        )

    thresholds = {
        "full_distribution": None,
        "up_to_p99_9": prices.quantile(0.999),
        "up_to_p99_5": prices.quantile(0.995),
        "up_to_p99": prices.quantile(0.99),
    }

    full_mean = float(prices.mean())
    results = []

    for scenario, threshold in thresholds.items():
        if threshold is None:
            restricted = prices
        else:
            restricted = prices[
                prices <= threshold
            ]

        excluded_count = len(prices) - len(restricted)

        mean_change_pct = (
            (restricted.mean() - full_mean)
            / full_mean
            * 100
        )

        results.append(
            {
                "scenario": scenario,
                "threshold_eur": threshold,
                "retained_count": len(restricted),
                "excluded_count": excluded_count,
                "retained_pct": (
                    len(restricted)
                    / len(prices)
                    * 100
                ),
                "mean_eur": restricted.mean(),
                "median_eur": restricted.median(),
                "standard_deviation_eur": restricted.std(),
                "maximum_eur": restricted.max(),
                "mean_change_pct_vs_full": mean_change_pct,
            }
        )

    return pd.DataFrame(results)


def write_csv(
    data: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> Path:
    """Write one evaluation table to CSV."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / filename
    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def main() -> None:
    """Run the baseline evaluation."""
    args = parse_arguments()

    listings = load_listings(
        args.listings
    )
    price_summary = build_price_summary(
        listings
    )
    price_sensitivity = build_price_sensitivity(
        listings
    )

    summary_path = write_csv(
        price_summary,
        args.output_dir,
        "price_summary.csv",
    )
    sensitivity_path = write_csv(
        price_sensitivity,
        args.output_dir,
        "price_sensitivity.csv",
    )

    print("Price evaluation complete.")
    print(
        "Non-missing prices: "
        f"{listings['price_eur'].notna().sum():,}"
    )
    print(
        "Median price: "
        f"€{listings['price_eur'].median():,.2f}"
    )
    print(
        "Mean price: "
        f"€{listings['price_eur'].mean():,.2f}"
    )
    print(
        "Maximum price: "
        f"€{listings['price_eur'].max():,.2f}"
    )
    print(
        f"Price summary written to: {summary_path.resolve()}"
    )
    print(
        "Sensitivity analysis written to: "
        f"{sensitivity_path.resolve()}"
    )


if __name__ == "__main__":
    main()
