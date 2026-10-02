# Data reference

Source: [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

| Normalized field | Meaning |
|---|---|
| pickup_datetime | Meter engagement time (`tpep_` for yellow, `lpep_` for green) |
| dropoff_datetime | Meter disengagement time |
| trip_distance | Meter-reported miles |
| fare_amount | Metered time-and-distance fare in dollars |
| passenger_count | Driver-entered passenger count; retained but not modeled |
| PULocationID / DOLocationID | TLC pickup/drop-off taxi-zone identifiers |
| payment_type | Payment code; used for descriptive analysis only |

| Code | Label | Interpretation |
|---|---|---|
| 0 | Flex Fare (current TLC definition) | Fare regime; not a card/cash method |
| 1 | Credit card | Payment method |
| 2 | Cash | Payment method |
| 3 | No charge | Payment status |
| 4 | Dispute | Payment status |
| 5 | Unknown (reported code) | Explicitly reported unknown |
| 6 | Voided trip | Payment status |
| Missing/null | Missing payment information | Not a payment code; not equivalent to 5 |
| Other | Unrecognized code | Retained and labeled separately |

The supplied older dictionaries list 1–6. The [March 2025 TLC yellow dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) and [green dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf) additionally define 0 as Flex Fare. That current definition is shown with a version caveat; it is not independently verified for every zero-coded record in 2020–2024. Zeros and green nulls were confirmed in the downloaded source files, rather than introduced by label assignment.

All categories remain in the accepted-trip analysis if the fare/distance/timestamp rules pass. Positive `fare_amount` means a positive recorded fare, not proof of payment settlement. Codes 3–6 are not automatically excluded. Payment type is descriptive and is excluded from model inputs. Card/cash-only model comparison is optional future work and has not been performed.

The local source dictionaries are `MetaData_data_dictionary_trip_records_yellow.pdf` (May 2022) and `Metadata_dictionary_trip_records_green.pdf` (May 2018). Cash tips are absent from `tip_amount` and `total_amount` according to these documents. `fare_amount` must not be described as the total paid.

Source metadata PDFs remain unchanged.
