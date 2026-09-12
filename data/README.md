This project uses the Inside Airbnb Sicily Detailed Listings dataset, snapshot 30 June 2026, downloaded on 7 September 2026.

The raw dataset is not stored in this repository. Place the source file locally at:

```text
data/raw/listings.csv.gz
```

The profiling script records the source filename, SHA-256 checksum, file
size, and dataset dimensions in
`outputs/profiling/dataset_metadata.csv`. This allows the exact input file
used for the analysis to be identified without storing the raw dataset in
the repository.
