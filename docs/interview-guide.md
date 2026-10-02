# Interview walkthrough

## Opening

“I built a reproducible Spark pipeline covering yellow and green NYC taxi records from 2020 through 2024. It normalizes monthly schemas, records quality checks, persists the full cleaned dataset, analyzes demand, and compares three completed-trip fare models. I use temporal validation and test periods, and I checked whether larger training samples improve validation performance.”

Use the README and saved notebook outputs for the exact measured counts and scores.

## Demonstration

1. Open the real-data notebook and show its five-year source coverage and quality counts.
2. Show full cleaned Parquet versus the explicitly bounded SQLite snapshot.
3. Explain two demand findings and a fare-distribution finding using the saved tables and charts.
4. Compare model validation scores, the larger-sample experiment and the untouched test result.
5. Show the reproduced 6,860.8-mile record and explain how training-only input caps changed the linear baseline. Discuss the late-2022 meter pricing change as a temporal-generalization challenge.
6. Use `nyc-taxi demo` if an interviewer wants a quick execution check without downloading data.

## Decisions to defend

- Parquet preserves types and compression; CSV is retained as an optional compatibility path.
- Requested monthly files must all be present, and typed name-based union prevents schema drift errors.
- Full-data engineering and EDA are separate from sampled modeling.
- Yellow and green fleets need different sampling fractions because their record volumes differ.
- Preprocessing is learned from training only, and validation selects candidates before test scoring.
- Extreme positive records remain in EDA; model input caps are learned on training data and checked against an uncapped linear comparison.
- The SQLite snapshot supports a portable demonstration without duplicating a very large dataset locally. Full export is configurable.
- Realized duration makes this a completed-trip estimator. A quote system would require planned-route distance and predicted duration with a different evaluation design.

## Limits

Do not claim a cluster benchmark, production deployment, full SQLite export, or exhaustive hyperparameter optimization. Discuss the measured local runtime, the fixed-seed experiment, and any remaining tail errors. Larger training samples are retained because of their measured validation performance, not because larger numbers automatically mean better models.
