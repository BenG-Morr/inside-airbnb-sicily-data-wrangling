# Inside Airbnb Sicily data wrangling

Reproducible Python pipeline for profiling, cleaning and tidying the Inside Airbnb Sicily Detailed Listings dataset, with documented data-quality decisions and validation.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # source file kept local; not committed
│   └── processed/           # generated refined datasets; not committed
├── outputs/
│   └── profiling/           # generated profiling diagnostics
├── src/
│   └── 01_profile.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python src/01_profile.py \
  --input data/raw/listings.csv.gz \
  --output-dir outputs/profiling
```

## Run profiling

Place `listings.csv.gz` in `data/raw/`, then run:

```bash
python src/01_profile.py \
  --input data/raw/listings.csv.gz \
  --output-dir outputs/profiling
```

The script currently generates a basic column-level profile containing data types, completeness statistics and uniqueness information.