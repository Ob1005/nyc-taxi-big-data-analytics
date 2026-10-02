# Methodology

## Question and population

Estimate `fare_amount` before travel, given the pickup and destination zones, and compare it with fare reconstruction after travel. The target is the recorded fare component, excluding separately recorded tips, tolls and surcharges. Yellow is the primary fleet; green provides a second comparison with the same procedure.

The source comprises every monthly yellow and green TLC Parquet file from 2020 through 2024. Each file is normalized separately before combining it, because numeric types vary between monthly files. All accepted trips contribute to exploratory analysis; only modelling uses a sample. Cleaned records stay in Parquet and are queried with Spark SQL. Small published tables are JSON.

Payment type is excluded from model inputs: it is not a reliable pre-trip attribute and describes payment method or status rather than trip geometry. A positive recorded fare does not establish that the trip was paid. No payment-code restriction is applied.

## Cleaning and first-failure accounting

All thresholds live in `src/nyc_taxi/config.py`. These are stated assumptions defining ordinary plausible taxi trips, not proof that every rejected record is erroneous. Unusual legitimate trips may be excluded. Rules execute in the order below; a row with several failures is counted only against its first failure.

| Order | Rule / reason | Why |
|---|---|---|
| 1 | Both timestamps present / `missing_timestamp` | Duration and temporal evaluation require timestamps. |
| 2 | Pickup year 2020–2024 / `outside_period` | Match the study population, including spillover dates in source files. |
| 3 | Distance finite and positive / `distance_missing_or_nonpositive` | Missing, invalid or nonpositive distance cannot describe the journey. Positive infinity fails the next rule. |
| 4 | Distance ≤100 miles / `distance_over_limit` | Exclude unusually long journeys and implausible distance records from this NYC fare study. |
| 5 | Duration ≥1 minute / `duration_too_short` | Avoid zero, reversed and near-zero timestamp intervals. |
| 6 | Duration ≤6 hours / `duration_too_long` | Exclude likely recording problems and journeys outside an ordinary taxi-trip scope. |
| 7 | Distance / duration ≤80 mph / `speed_over_limit` | A sustained average above this limit is implausible for ordinary NYC taxi travel. |
| 8 | Fare finite and positive / `fare_missing_or_nonpositive` | Define a positive-recorded-fare population; missing and invalid values are unusable targets. Positive infinity fails the next rule. |
| 9 | Fare ≤$500 / `fare_over_limit` | Exclude exceptionally priced trips from the chosen ordinary-fare population. |

Duration is elapsed seconds divided by 60. Year, hour, weekday, month and pricing boundaries use New York local time. Boundary values are included. Rejection counts are order-dependent and are not independent estimates of each problem's prevalence. The run checks the combined counts against source-file footer row totals. [Results](results/README.md) reports every reason, including zero counts, and the source example that motivated inspecting implausible distances.

## Exploratory analysis

Four Spark SQL queries summarize accepted pickups by month, pickups by hour, mean fares by distance band, and highest mean fares by destination zone. The destination ranking requires at least 500 trips and restricts IDs to mapped zones 1–263, excluding unknown/outside codes. The ranking is descriptive and does not control for route length. Demand comparisons begin in 2020; there is no pre-pandemic comparison year. Green uses the same queries but receives shorter treatment in the walkthrough.

## Sampling and temporal evaluation

A seeded Bernoulli sample uses 0.2% of accepted yellow trips and 5% of accepted green trips by default. The larger green fraction compensates for its smaller fleet. These fractions are practical local-machine compute choices, not optimized statistical thresholds. Actual sample sizes vary; every split count is reported. Both feature settings and the mean baseline use exactly the same sampled rows within a fleet.

Train on pickups in 2020–2022, compare candidates on 2023, and evaluate each selected setting once on 2024. Temporal splitting better matches use on future trips than a random split, but supplies only one historical holdout. Test results do not guide candidate selection. Models are fitted on training only, without refitting on validation. The baseline predicts the training mean fare.

## Information available to each setting

Both settings contain pickup and destination zone, pickup hour, weekday and month, a known pricing-regime flag, and historical route distance/duration medians. Pre-trip assumes the destination has already been supplied; it is not a destination predictor. Route medians are computed from the training sample only with Spark's approximate median aggregation, grouped by pickup/destination pair. Missing or unseen pairs use the overall training-sample distance and duration medians. The fitted lookup and fallback values are saved with the model. Encoding vocabularies are also fitted only on training; unseen categories use an unknown category.

The pre-trip model never uses that trip's realized distance or duration. The post-trip setting adds both, providing a reconstruction upper bound. Historical route medians are kept fixed for validation and test; no later outcomes enter them. Training features use training route summaries, a conventional in-sample aggregate; no training-score claim is made.

The pricing flag is true for pickups on or after December 19, 2022, the effective date in the [TLC fare-change notice](https://www.nyc.gov/assets/tlc/downloads/pdf/industry-notices/industry_notice_22_02_english.pdf). This is known external information, not a learned change point. The notice ties use of the new tariff to meter recalibration, so the date flag is a pricing-period proxy rather than evidence that every meter changed that day. Only the final days of the training period reflect the new regime, so its coverage is limited. The flag allows a model to represent a regime difference; without an ablation or causal design, no isolated accuracy gain or causal effect is claimed.

## Models and errors

Linear regression is a simple additive reference; random forest and gradient-boosted trees allow nonlinear interactions. Each feature setting selects the lowest validation RMSE among these three candidates. Hour, weekday and month are kept as integer inputs for a simple shared representation; the linear model cannot represent cyclic time patterns as flexibly as the trees. Modest fixed tree settings limit compute and keep the comparison explainable; there is no hyperparameter search. RMSE emphasizes larger misses. Test MAE, RMSE, R² and the percentages within $2 and $5 describe complementary aspects of performance. Predictions are not rounded or forced positive, so these metrics expose errors as produced by the fitted model.

Error by fare band and the 20 largest absolute test residuals are saved after filtering. They help identify expensive-trip failures that an overall average can conceal; they do not change model selection or the evaluated population.

## Execution limits

Spark runs on one machine with two local worker threads and a 4 GB driver. Full-data cleaning and SQL aggregation stream through Parquet; only the modelling sample and transformed sample are cached. Runtime and environment are measured by each run, not extrapolated to a cluster. This is an analysis pipeline, not a deployed estimator.
