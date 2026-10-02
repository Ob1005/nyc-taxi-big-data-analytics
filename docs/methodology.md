# Methodology

## Prediction question

Estimate the metered time-and-distance fare for a completed trip. The distance and duration are realized measurements, not information available when booking. The target is `fare_amount`, excluding tips and surcharges. This is an analytical estimator, not a pre-trip quote service.

## Coverage and data quality

The main experiment reads 60 monthly files per fleet, January 2020 through December 2024. Missing requested files cause the run to fail rather than silently analyze a subset. Each monthly file is projected onto a typed common schema before name-based union, avoiding positional errors from schema drift. The efficient default uses Parquet. Optional CSV conversion produces partitioned CSV directories and retains the originals; it does not reduce disk use.

Reject missing timestamps, pickup years outside 2020–2024, nonfinite/nonpositive distance or fare, and nonpositive duration. Each record receives one first-failure reason, so quality counts reconcile with the raw source count. Missing passenger counts are retained because that driver-entered field is not used for modeling. The population is trips with positive recorded fares, not confirmed paid trips. A no-charge or disputed trip can still have a positive recorded fare. Payment codes 0 and 3–6 and missing payment information are retained when the other rules pass. Zero/negative recorded fares are rejected irrespective of payment code.

Extreme positive values are retained in the cleaned data and EDA. They are investigated through distribution summaries and worst-residual exports. For modeling, distance and duration inputs are capped at training-set 99.9th percentiles. Raw measurements and fare labels remain unchanged. Caps are learned only from training records; they are applied to all later data using the saved preprocessing pipeline. An uncapped linear-regression comparison uses the same base training and validation records to assess sensitivity to this decision.

## Storage

All accepted records are saved as cleaned Parquet, which is the full analytical dataset. Full-data frames are scanned rather than cached, avoiding large driver/disk spill requirements. Only sampled modeling records are cached.

By default, SQLite contains a **bounded snapshot of at most 100,000 accepted rows per fleet**, for portable SQL inspection. This snapshot is not a representative statistical sample and is not used to train models or generate EDA. It is explicitly labeled in `run.json`. `--sqlite-limit 0` requests every accepted row, subject to a conservative disk-headroom check. This differs from the original full SQLite export to keep a local five-year run practical on limited storage.

SQLite repartitions to approximately 50,000 rows per partition before streaming 10,000-row insert batches. It validates row counts and integrity before atomically replacing the database. This bounds driver transfer memory, but full SQLite export remains a serial bottleneck. Timestamps are formatted in Spark in New York local time before serialization. Source timestamps lack an explicit offset, so daylight-saving ambiguity remains a limitation.

## EDA definitions

Distance intervals are [0,1), [1,2), [2,5), [5,10), [10,20), and [20,infinity) miles; zero distances have already been rejected. Long trips are strictly greater than 10 miles. Weekend days are Saturday and Sunday.

Highest-average-fare drop-off rankings require at least 500 accepted trips and valid TLC zone IDs. Counts accompany averages to make their support visible. Zone IDs remain categorical identifiers; joining neighborhood names is a future extension. Weekday/weekend results include totals and trips per observed date; zero-trip dates are not included, so this is not full calendar normalization.

## Modeling and sample size

The starting model sample fractions reproduce the original choices: 0.2% yellow and 5% green. They apply to modeling only, not to data cleaning or EDA. A fixed seed defines the sample. Training uses 2020–2022, validation 2023 and test 2024.

Sample-size sensitivity is evaluated using a pool twice as large. The validation and test samples stay fixed while training grows. Three regressors are compared on the base training sample: regularized linear regression, random forest and gradient-boosted trees. The lowest validation-RMSE model is then fitted on the larger training sample with preprocessing relearned on that larger sample. The larger candidate is retained only if validation RMSE improves. This is a limited learning-curve comparison, not an exhaustive tuning search. The test sample is scored only after choosing the final candidate.

Features include realized distance and duration, hour, weekday, month, pickup zone, drop-off zone and distance bin. Training-only `StringIndexer` and one-hot encoders handle categorical values, including unseen categories, before `VectorAssembler`. Payment method, final rate code, total amount, tips and fare-derived fields are excluded from inputs. Time fields are numeric; cyclical encoding has not been evaluated.

## Evaluation and diagnostics

Report validation and held-out RMSE, MAE and R², together with a training-mean baseline. Diagnostics include median/p90/p99 absolute errors, errors by fare band, and the 20 worst residuals. The share of squared error contributed by those 20 records helps distinguish broad poor performance from isolated extreme errors.

Prediction plots use a seeded hash ranking over trip fields to choose up to 2,000 records across the held-out dataset, avoiding bias toward the first file or partition. Exact duplicate trips receive the same hash and tied selection is not guaranteed stable. Both full-range and $0–100 views are shown; reported metrics always cover the entire held-out sample.

## Runtime and reproducibility

`experiment.json` records local environment details and wall-clock processing time. Each `run.json` breaks down load/clean/Parquet, SQLite snapshot, EDA and modeling time. Downloads and notebook presentation are excluded from the main processing timer. This is a measured local run, not a clustered benchmark.

Source filenames, row counts and byte sizes are recorded in the download manifest; this is not a cryptographic content lock. Upstream files may change. Experiments are written to separate output roots when preservation is needed; cleaned data directories use overwrite semantics. Published compact results include metrics and aggregates, while large data, database and model binaries are excluded from Git.

## Pricing change near the temporal split

The [official TLC notice](https://www.nyc.gov/assets/tlc/downloads/pdf/industry-notices/industry_notice_22_02_english.pdf) establishes a meter fare change effective 19 December 2022. Most training trips precede it, while validation/test trips follow it. The model has no explicit pricing-regime feature. Underestimation in the plots is consistent with this shift, but its contribution has not been isolated experimentally. This is a meaningful temporal-generalization limitation, not a reason to revert to random testing that mixes periods.

## Payment categories

The payment table preserves each original code and labels its meaning. Card/cash are payment methods; no charge/dispute/voided are payment statuses; code zero denotes Flex Fare in the current TLC dictionary. Reported unknown (5), missing information and unrecognized values remain separate. The supplied 2018/2022 dictionaries omit zero; the March 2025 TLC mapping is cited explicitly and is not independently verified for every historical zero-coded record.

Payment-method conclusions compare explicitly coded card/cash records; status and missingness counts are shown alongside them without assuming a method for those trips. These records still contribute to EDA and the existing model population. No payment-based exclusions or model retraining were introduced. A card/cash-only sensitivity comparison is optional future work.
