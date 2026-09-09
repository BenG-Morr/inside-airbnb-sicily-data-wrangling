# Inside Airbnb Sicily data wrangling

Reproducible Python pipeline for profiling, cleaning and tidying the
Inside Airbnb Sicily Detailed Listings dataset, with documented
data-quality decisions and validation.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # source data kept local; not committed
│   └── processed/           # generated refined datasets; not committed
├── outputs/
│   └── profiling/           # generated profiling diagnostics
├── src/
│   └── 01_profile.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run profiling

Place `listings.csv.gz` in `data/raw/`, then run:

```bash
python src/01_profile.py \
  --input data/raw/listings.csv.gz \
  --output-dir outputs/profiling
```

The profiling script generates:

- a column-level profile of data types, completeness and uniqueness;
- price representation, missingness and distribution diagnostics;
- review-score missingness checks;
- bathroom completeness and consistency checks;
- host-field consistency checks;
- amenities structure checks;
- date and boolean representation checks;
- records for manual inspection of the highest observed prices; and
- a consolidated data-quality inventory with proposed wrangling treatments.

The raw dataset is not modified by the profiling process.