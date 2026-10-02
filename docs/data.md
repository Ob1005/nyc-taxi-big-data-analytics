# Data

Source: [NYC TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), monthly yellow and green trip records for 2020–2024.

`nyc-taxi download` fetches the monthly Parquet files and checks their footers. Existing valid files are reused. Raw files and cleaned Parquet are local and excluded from Git because of their size. A full run requires every monthly file for both fleets.

The pipeline projects pickup/drop-off timestamps, pickup/drop-off zone IDs, distance and recorded fare into consistent types. The supplied yellow and green metadata PDFs remain in the repository as source references. [Methodology](methodology.md) describes the analysis population and feature availability. [Results](results/README.md) reports source totals and exclusions from the actual run.
