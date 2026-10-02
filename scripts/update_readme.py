"""Write the portfolio overview from the measured five-year experiment."""
import json
from pathlib import Path

experiment=json.loads(Path('docs/results/experiment.json').read_text())
runs=experiment['fleets']
metrics=['| Fleet | Selected model | Training rows | Test MAE | Test RMSE | Test R² |',
         '|---|---|---:|---:|---:|---:|']
for fleet,run in runs.items():
    m=run['modeling']
    metrics.append(f"| {fleet.title()} | {m['selected_model']} | {m['selected_training_rows']:,} | ${m['test']['mae']:.2f} | ${m['test']['rmse']:.2f} | {m['test']['r2']:.3f} |")
coverage=', '.join(f"**{run['full_cleaned_rows']:,} {fleet} trips**" for fleet,run in runs.items())
text=f'''# NYC Taxi Fare Analytics

**A reproducible five-year Spark project that processes NYC yellow and green taxi records, analyzes demand and estimates completed-trip metered fares.**

Python · PySpark · Spark SQL · Spark ML · Parquet · SQLite · Matplotlib

## Measured five-year results

The verified local experiment reads **all 120 monthly TLC files from 2020 through 2024** and retains {coverage}. Cleaning and exploratory analysis use all accepted records. Modeling starts with a 0.2% yellow sample and a 5% green sample, then checks whether larger training samples improve validation performance.

Training covers 2020–2022, validation 2023, and the untouched test period 2024. The final models were selected using validation RMSE.

{chr(10).join(metrics)}

Processing took **{experiment['elapsed_seconds']/60:.1f} minutes** on the recorded local environment, excluding downloads and notebook presentation. These are completed-trip fare estimates, using realized distance and duration, not booking quotes. See [results, diagnostics and limitations](docs/results/README.md) for exact split counts, baseline comparisons, sample-size sensitivity, model failures and runtime breakdowns.

![Yellow held-out fare estimates](docs/results/yellow/actual_vs_predicted.png)

## Explore the project

Start with [the real-data notebook](notebooks/nyc_taxi_portfolio.ipynb). It includes saved outputs for both fleets and six practical stages: ingestion, cleaning, persistence, exploratory analysis, feature engineering and regression modeling.

- [Methodology](docs/methodology.md): population, feature choices, temporal splits and limits.
- [Verified results](docs/results/README.md): compact evidence from the complete local run.
- [Data reference](docs/data.md): source fields and payment codes.
- [Interview walkthrough](docs/interview-guide.md): decisions and results to explain.

The reusable implementation lives in `src/nyc_taxi/`, with meaningful checks in `tests/`.

## Run the notebook

Use Python 3.9–3.12 and Java 17. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m jupyterlab
```

Open `notebooks/nyc_taxi_portfolio.ipynb` and run its cells in order. By default it reads the published real results, so a clone does not need the large datasets. Set `RECOMPUTE = True` in its setup cell to download and run the full experiment again. The root `newyork-taxi.ipynb` is an identical convenience copy.

## Quick execution check

```bash
nyc-taxi demo
pytest -q
```

The demo uses clearly labeled synthetic records for both fleets. It checks execution and produces outputs in `outputs/demo/`; its scores are not portfolio performance evidence.

## Reproduce the full experiment

```bash
nyc-taxi download
nyc-taxi run --output outputs/full-five-year
python -m nyc_taxi.reporting --run outputs/full-five-year/real
python scripts/update_readme.py
```

Default starting samples are `--yellow-sample-fraction 0.002` and `--green-sample-fraction 0.05`. The selected model family is also tried with twice as much training data against the same validation sample. Use `--no-sample-comparison` to run only the starting experiment. `--sample-fraction` optionally overrides both fleets; it should be chosen deliberately.

For the separate reproduced error investigation, run `python -m nyc_taxi.investigate` before publishing results.

Changing sampling or input coverage produces a new experiment. Refresh the saved notebook outputs after publishing new results; published metrics should always describe the actual run.

## Storage and practical scope

All accepted records are saved in `data/cleaned/real/` as compressed Parquet. Modeling and EDA read that full dataset. The default SQLite export is an explicitly bounded **100,000-row snapshot per fleet** for portable SQL inspection. It is not a representative model sample. `run.json` records its exact count; `--sqlite-limit 0` requests a full export if disk space permits.

Parquet is the efficient default. `--csv` converts each monthly source to a directory of CSV part files and reads it back with explicit casts. It retains the Parquet originals and uses additional disk space. Converting to CSV is a compatibility option, not a storage-saving step.

Large input/output artifacts are excluded from Git. Downloads and processed data require local disk space; full SQLite export duplicates a substantial dataset and serializes rows through the driver. The measured experiment uses a Parquet warehouse plus SQLite snapshots, not full SQLite copies.

## Pipeline

```mermaid
flowchart LR
    A[120 monthly TLC Parquet files] --> B[Typed normalization and quality accounting]
    B --> C[Full cleaned Parquet]
    C --> D[Bounded SQLite snapshots]
    C --> E[Full-data Spark SQL and EDA]
    C --> F[Separate fleet samples]
    F --> G[2020–2022 training / 2023 validation / 2024 test]
    G --> H[Training-only caps and categorical encoding]
    H --> I[Three regressors and sample-size comparison]
    I --> J[Held-out metrics and residual diagnostics]
```

## What the project demonstrates

- Validated concurrent downloads, retries and atomic file publication.
- Complete requested source coverage, typed schemas and name-based unions.
- Reconciled first-failure cleaning counts and retained raw outlier measurements.
- Full-data processing without caching the entire dataset locally.
- Nine EDA outputs per fleet, with a 500-trip minimum for fare-ranking zones.
- Training-only preprocessing, separate validation/test data and a mean baseline.
- A measured training-size comparison and an uncapped linear-regression sensitivity check.
- Seeded prediction previews, errors by fare band and worst-residual exports.
- Recorded environment, stage timings, verified SQLite snapshots and saved model artifacts.

## Interpretation and limitations

The target is `fare_amount`, excluding tips and surcharges. Distance and duration are realized values. Cleaning defines a positive-recorded-fare population, not confirmed paid trips. No-charge/dispute/unknown codes and missing payment information remain if other rules pass; payment type is excluded from model inputs. Payment tables label methods, statuses and missingness separately. Training-only distance/duration caps can suppress real extreme behavior, so uncapped comparisons and tail diagnostics accompany the headline scores.

These are fixed-seed results using modest model settings and one temporal holdout. They do not establish deployment readiness, repeated-sample uncertainty or cluster-scale performance. Extreme recorded fares remain visible in EDA and model labels; read the error investigation before interpreting RMSE alone.

## Source and verification

[NYC Taxi & Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). TLC provides monthly records from authorized technology providers and does not guarantee their accuracy. The supplied green/yellow dictionaries describe field meanings but do not establish an identical schema for every month.

Six tests cover malformed records, timestamp preservation, SQL replacement safety, CSV/schema drift, training-only caps/unseen categories and missing requested files. Both fleet demos and the real five-year experiment are executed locally. The repository workflow runs checks and the synthetic demo on GitHub.
'''
Path('README.md').write_text(text)
