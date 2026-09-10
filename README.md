# Inside Airbnb Sicily data wrangling

Reproducible Python pipeline for profiling, cleaning, structurally tidying,
validating and evaluating the Inside Airbnb Sicily Detailed Listings
dataset.

The project focuses on improving structural tidiness and representational
consistency for descriptive analysis while avoiding unnecessary information
loss or unsupported correction of source values.

## Repository structure

```text
.
├── data/
│   ├── README.md
│   ├── raw/                         # source data kept local; not committed
│   └── processed/                   # generated refined data; not committed
├── outputs/
│   ├── profiling/                   # profiling and data-quality diagnostics
│   └── evaluation/                  # post-wrangling evaluation outputs
├── src/
│   ├── 01_profile.py                # initial data-quality assessment
│   ├── 02_clean.py                  # cleaning and structural tidying
│   ├── 03_validate.py               # post-wrangling validation
│   └── 04_evaluate.py               # statistical evaluation
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

The project uses pandas and NumPy. The version ranges are specified in
`requirements.txt`.

## Input data

Place the Inside Airbnb Sicily Detailed Listings file at:

```text
data/raw/listings.csv.gz
```

The raw and processed datasets are excluded from version control.

During profiling, the source filename, file size, dimensions and SHA-256
checksum are recorded in:

```text
outputs/profiling/dataset_metadata.csv
```

This identifies the exact raw dataset snapshot used for the analysis
without storing the source file in the repository.

## Pipeline

The scripts are intended to be run in numerical order.

### 1. Profile the raw dataset

```bash
python src/01_profile.py \
  --input data/raw/listings.csv.gz \
  --output-dir outputs/profiling
```

The profiling stage assesses column completeness and uniqueness, price
representation and missingness, price-quote structure, review-score
missingness, bathroom consistency, host-field consistency, nested
amenities, date and Boolean representations, duplicate records and the
distribution of observed prices.

It also produces a consolidated data-quality inventory documenting the
identified issues, associated risks, proposed treatments and validation
requirements.

### 2. Clean and structurally tidy the data

```bash
python src/02_clean.py \
  --input data/raw/listings.csv.gz \
  --output data/processed/listings_clean.csv \
  --hosts-output data/processed/hosts.csv \
  --amenities-output data/processed/listing_amenities.csv \
  --price-quotes-output data/processed/price_quotes.csv \
  --price-quote-line-items-output \
  data/processed/price_quote_line_items.csv
```

The cleaning stage derives a numeric EUR price field while preserving the
original display value, harmonises selected date and Boolean
representations, derives missing bathroom counts where supported by the
textual representation, and flags conflicting bathroom values rather than
overwriting them.

Repeated host attributes are separated into a host-level table, nested
amenities are transformed into listing–amenity relationships, and nested
price-quote information is separated into quote-level and line-item
tables. Fields that are entirely empty in the analysed snapshot are
removed only after their complete missingness has been revalidated.

The generated processed tables are:

```text
data/processed/listings_clean.csv
data/processed/hosts.csv
data/processed/listing_amenities.csv
data/processed/price_quotes.csv
data/processed/price_quote_line_items.csv
```

### 3. Validate the processed data

```bash
python src/03_validate.py \
  --raw data/raw/listings.csv.gz \
  --listings data/processed/listings_clean.csv \
  --hosts data/processed/hosts.csv \
  --amenities data/processed/listing_amenities.csv \
  --price-quotes data/processed/price_quotes.csv \
  --price-quote-line-items \
  data/processed/price_quote_line_items.csv
```

The validation stage checks primary-key uniqueness, referential integrity,
row preservation, selected cleaning outcomes, date and Boolean
harmonisation, structural review-score missingness, exclusion of verified
empty fields, and preservation of the normalised host, amenity and
price-quote structures.

### 4. Evaluate the refined price data

```bash
python src/04_evaluate.py \
  --listings data/processed/listings_clean.csv \
  --output-dir outputs/evaluation
```

The evaluation stage produces descriptive price statistics and a
sensitivity analysis of the extreme upper tail.

Extreme prices are retained in the refined dataset because their magnitude
alone does not establish that they are erroneous. Instead, the sensitivity
analysis compares the full distribution with scenarios restricted at the
99th, 99.5th and 99.9th percentiles to quantify their influence on
descriptive statistics without altering the cleaned data.

The principal evaluation outputs are:

```text
outputs/evaluation/price_summary.csv
outputs/evaluation/price_sensitivity.csv
```

## Reproducibility

The raw source is treated as immutable. Profiling diagnostics and evaluation
results are generated by the scripts and retained in the repository, while
raw and processed datasets remain local.

Each transformation is accompanied by explicit validation checks so that
unexpected source structures, conflicting values or information loss cause
the pipeline to fail rather than being silently accepted.