"""Build the real-results walkthrough; execution saves verified outputs separately."""
import json
from pathlib import Path

cells=[]

def md(source):
    cells.append({'cell_type':'markdown','metadata':{},'source':source.splitlines(True)})

def code(source):
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(True)})

md('''# NYC Taxi Fare Analytics — 2020–2024

A real-data walkthrough of yellow and green taxi ingestion, cleaning, storage, exploratory analysis, feature engineering and Spark regression. Saved outputs come from the verified five-year local experiment.

Run cells in order. The default reads compact published results, so it also works after cloning without downloading the full dataset. Set `RECOMPUTE = True` in the next cell to download all 120 monthly files and rerun the full pipeline. This takes substantially longer than reading the results.

The target is **completed-trip metered fare**, using realized distance and duration. It is not a pre-trip quote or total passenger spending. The quick synthetic demo remains available separately with `nyc-taxi demo`.
''')
code('''from pathlib import Path
import csv
import json
import subprocess
import sys
import os
from IPython.display import Markdown, display, Image

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
RECOMPUTE = False
RUN_ROOT = ROOT / "outputs/full-five-year/real"
RESULTS = ROOT / "docs/results"

if RECOMPUTE:
    subprocess.run([sys.executable, "-m", "nyc_taxi.cli", "download"], check=True)
    subprocess.run([sys.executable, "-m", "nyc_taxi.cli", "run", "--output", "outputs/full-five-year"], check=True)
    subprocess.run([sys.executable, "-m", "nyc_taxi.reporting", "--run", str(RUN_ROOT)], check=True)

experiment = json.loads((RESULTS / "experiment.json").read_text())
fleets = experiment["fleets"]

def table(records, columns=None):
    if not records:
        display(Markdown("No records meet this analysis's criteria."))
        return
    columns = columns or list(records[0])
    def formatted(value, column):
        if column in ("year", "month", "distance_bin", "payment_type"):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.4f}"
        try:
            if str(value).isdigit():
                return f"{int(value):,}"
            return f"{float(value):,.4f}"
        except (ValueError, TypeError):
            return str(value)
    content = "| " + " | ".join(columns) + " |\\n"
    content += "| " + " | ".join("---" for _ in columns) + " |\\n"
    for record in records:
        content += "| " + " | ".join(formatted(record.get(c, ""), c) for c in columns) + " |\\n"
    display(Markdown(content))

def csv_rows(fleet, name):
    with (RESULTS / fleet / name).open() as stream:
        return list(csv.DictReader(stream))

print("Real TLC data; fleets:", ", ".join(fleets))
print("Years:", fleets["yellow"]["years"])
''')
md('''## 1. Ingestion and source coverage

The downloader retries network failures, checks Parquet metadata, and publishes files atomically. All requested months must be present. The main run uses compressed Parquet; CSV conversion is optional and uses additional space.

The manifest provides provenance for the actual inputs, including filenames, byte sizes and row counts. It is not a cryptographic content lock.
''')
code('''manifest = json.loads((RESULTS / "download-manifest.json").read_text())
coverage = []
for fleet, run in fleets.items():
    source_files = run["source_files"]
    coverage.append({"fleet": fleet, "monthly_files": len(source_files),
                     "raw_rows": f"{sum(item['rows'] for item in source_files):,}",
                     "first_file": source_files[0]["file"], "last_file": source_files[-1]["file"]})
table(coverage)
assert all(row["monthly_files"] == 60 for row in coverage)
assert len(manifest) == 120
''')
md('''## 2. Cleaning and schema normalization

Yellow uses `tpep_*` timestamps and green uses `lpep_*`. Each monthly file is projected onto the same typed schema before a name-based union. Cleaning rejects missing timestamps, pickup years outside the requested period, nonfinite/nonpositive fare or distance, and nonpositive duration.

Each row has one first-failure reason, so accepted plus rejected counts reconcile to the source total. Passenger counts are retained even when missing because they are not a model input. Extreme positive measurements remain in the cleaned data for inspection.
''')
code('''quality_rows = []
for fleet, run in fleets.items():
    for reason, count in sorted(run["quality"].items()):
        quality_rows.append({"fleet": fleet, "reason": reason, "rows": f"{count:,}"})
    assert sum(run["quality"].values()) == sum(source["rows"] for source in run["source_files"])
table(quality_rows)
''')
md('''## 3. Full storage and portable SQLite inspection

Every accepted trip is saved in cleaned Parquet. SQLite contains an explicitly bounded snapshot of at most 100,000 accepted rows per fleet, which is verified before replacing the database. This saves local storage and supports portable SQL inspection. The snapshot is not a representative modeling sample and is not used for EDA.

For an optional full SQLite export, use `--sqlite-limit 0` with sufficient disk space. The main experiment's measured scope is recorded below.
''')
code('''table([{"fleet": fleet, "full_cleaned_rows": f"{run['full_cleaned_rows']:,}",
        "sqlite_rows": f"{run['sqlite_rows']:,}", "sqlite_scope": run["sqlite_scope"]}
       for fleet, run in fleets.items()])

import sqlite3
for fleet in fleets:
    database = RUN_ROOT / fleet / f"nyc_{fleet}_taxi.db"
    if database.exists():
        with sqlite3.connect(database) as connection:
            count = connection.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        assert count == fleets[fleet]["sqlite_rows"]
        assert integrity == "ok"
        print(f"{fleet}: SQL count={count:,}, integrity={integrity}")
    else:
        print(f"{fleet}: SQLite binary is excluded from Git; its verified scope is recorded above.")
''')
md('''## 4. Exploratory analysis on all accepted trips

Spark SQL and DataFrame aggregations cover pickup zones, hour, month, distance-bin fares, highest-fare drop-offs, weekdays/weekends, trips over 10 miles, payment methods and fare distribution.

Drop-off rankings require at least 500 trips. Weekday/weekend averages use observed dates, not a complete calendar. The summaries describe trips with positive recorded fares, not confirmed settled payments. Codes 0 and 3–6 and missing payment information remain when other cleaning rules pass.
''')
code('''for fleet in fleets:
    display(Markdown(f"### {fleet.title()} demand"))
    display(Image(filename=str(RESULTS / fleet / "demand.png")))
    display(Markdown("**Average metered fare by distance interval**"))
    order = {"0-1": 0, "1-2": 1, "2-5": 2, "5-10": 3, "10-20": 4, "20+": 5}
    table(sorted(csv_rows(fleet, "fare_by_distance.csv"), key=lambda r: order[r["distance_bin"]]))
    display(Markdown("**Highest average fare drop-offs (at least 500 trips)**"))
    table(csv_rows(fleet, "highest_fare_dropoffs.csv"))
    display(Markdown("**Weekday/weekend comparison**"))
    table(csv_rows(fleet, "weekend_weekday.csv"))
''')
code('''for fleet in fleets:
    hourly = csv_rows(fleet, "hourly_trips.csv")
    busiest = max(hourly, key=lambda row: int(row["count"]))
    fare = {row["summary"]: row["fare_amount"] for row in csv_rows(fleet, "fare_distribution.csv")}
    monthly = csv_rows(fleet, "monthly_trips.csv")
    annual = {}
    for row in monthly:
        year = row["month"][:4]
        annual[year] = annual.get(year, 0) + int(row["count"])
    display(Markdown(f"**{fleet.title()} findings:** the busiest pickup hour was {int(busiest['pickup_hour']):02d}:00. "
                     f"Median metered fare was ${float(fare['50%']):.2f}, compared with a maximum of ${float(fare['max']):,.2f}. "
                     "The gap motivates examining tail errors rather than relying only on an average score."))
    table([{"year": year, "accepted_trips": f"{count:,}"} for year, count in sorted(annual.items())])
    display(Markdown("**Payment categories: methods, statuses and missing information**"))
    table(csv_rows(fleet, "payment_methods.csv"))
    display(Markdown("**Trips longer than 10 miles**"))
    table(csv_rows(fleet, "long_trip_fares.csv"))
''')
md('''### Interpreting payment categories

Card (1) and cash (2) identify payment methods. No charge (3), dispute (4) and voided trip (6) describe status. Reported unknown (5) is separate from missing information.

The supplied older dictionaries omit zero. The [March 2025 TLC dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) defines **0 as Flex Fare**; this is labeled as the current definition, without claiming it was independently verified for every historical record. The zeros and green nulls are present in the raw source files.

A positive recorded fare does not establish payment settlement. All these categories remain if the other cleaning rules pass, and payment type is not a model input. The existing model results are unchanged; a card/cash-only model comparison is optional future work.
''')
md('''## 5. Feature engineering and training-only preprocessing

The model uses realized distance/duration, pickup hour, weekday, month, pickup zone, drop-off zone and distance bin. Location IDs and distance bins are one-hot encoded. Encoders and input caps are learned from training data only and then saved for later predictions.

Distance and duration inputs are capped at training-set 99.9th percentiles to assess sensitivity to extreme measurements. Raw columns and target fares remain unchanged. Payment method, final rate code, tips and total amount are excluded from inputs.

The preview below comes from the bounded SQLite snapshot and illustrates feature definitions; it does not describe the modeling sample distribution.
''')
code('''from nyc_taxi.pipeline import NUMERIC, CATEGORICAL
print("Numeric feature definitions:", NUMERIC)
print("Categorical feature definitions:", CATEGORICAL)
for fleet, run in fleets.items():
    display(Markdown(f"### {fleet.title()} feature preview"))
    table(csv_rows(fleet, "feature_preview.csv"),
          ["trip_distance", "duration_minutes", "pickup_hour", "pickup_weekday", "pickup_month", "distance_bin"])
    print("Training-learned caps:", run["modeling"]["training_input_caps_p999"])
''')
md('''## 6. Train, compare sample sizes, and evaluate

Training uses 2020–2022, validation 2023, and test 2024. Starting sample fractions are 0.2% yellow and 5% green. Linear regression, random forest and gradient-boosted trees are compared using validation RMSE.

The selected regression family is then tried with a larger training sample, holding the validation/test samples fixed. The larger candidate is retained only if validation RMSE improves. The final candidate is evaluated once on the held-out test sample, with a training-mean baseline.
''')
code('''table([{"fleet": fleet, **run["modeling"]["split_counts"],
        "selected_training": run["modeling"]["selected_training_rows"]}
       for fleet, run in fleets.items()])
for fleet, run in fleets.items():
    m = run["modeling"]
    display(Markdown(f"### {fleet.title()} validation comparison"))
    table(m["validation"], ["model", "rmse", "mae", "r2"])
    display(Markdown("**Training-size sensitivity on the same validation records**"))
    table([{"training_rows": candidate["training_rows"], "training_fraction": candidate["training_fraction"],
            **candidate["validation"]} for candidate in m["sample_size_comparison"]])
''')
code('''table([{"fleet": fleet, "selected_model": run["modeling"]["selected_model"],
        **run["modeling"]["test"], "baseline_rmse": run["modeling"]["mean_baseline_test"]["rmse"]}
       for fleet, run in fleets.items()])
for fleet in fleets:
    display(Image(filename=str(RESULTS / fleet / "actual_vs_predicted.png")))
''')
md('''## Error analysis

An uncapped linear-regression comparison uses the same base training and validation records. Compare its scores and worst residuals with capped inputs. The difference measures sensitivity to input treatment, without proving that every extreme record is invalid.

Prediction plots use up to 2,000 seeded hash-selected test records. Their zoomed panel shows typical fares; metrics include the full test sample. Errors by fare band show whether large fares remain difficult.
''')
code('''for fleet, run in fleets.items():
    m = run["modeling"]
    capped = next(result for result in m["validation"] if result["model"] == "LinearRegression")
    display(Markdown(f"### {fleet.title()} linear-regression sensitivity"))
    table([{"variant": "uncapped", **m["uncapped_linear_validation"]},
           {"variant": "training-only input caps", **{key: capped[key] for key in ("rmse", "mae", "r2")}}])
    print("Share of uncapped squared error from the 20 worst records:",
          f"{m['uncapped_linear_diagnostics']['worst_20_share_of_squared_error']:.1%}")
    table(csv_rows(fleet, "validation_linear_uncapped_worst_errors.csv")[:5])
    display(Markdown("**Held-out errors by actual fare band**"))
    table(csv_rows(fleet, "test_selected_error_by_fare.csv"))
''')
md('''## Reproduced verification failure

A separate three-month diagnostic reproduces the earlier yellow linear-regression failure. It uses the same 28,888 training records and 29,150 validation records, comparing uncapped and training-capped feature inputs. This explains a pipeline weakness; it is not the main five-year performance result.
''')
code('''diagnostic_root = RESULTS / "verification-error-investigation"
if (diagnostic_root / "investigation.json").exists():
    investigation = json.loads((diagnostic_root / "investigation.json").read_text())
    table([{"variant": name, **value["metrics"]} for name, value in investigation["variants"].items()])
    with (diagnostic_root / "uncapped_worst_errors.csv").open() as stream:
        worst_records = list(csv.DictReader(stream))
    table(worst_records[:3])
    display(Markdown("The leading record reports **6,860.8 miles in about 39 minutes** for a **$32.50 fare**. "
                     "Uncapped linear extrapolation predicted roughly **$18,057**. "
                     "On the same train/validation records, training-learned input caps reduced RMSE from **$105.71 to $5.32**. "
                     "This illustrates why both typical errors and extreme residuals need inspection."))
else:
    print("Optional diagnostic evidence is absent; reproduce with python -m nyc_taxi.investigate.")
''')
md('''## Temporal pricing context

[TLC's official notice](https://www.nyc.gov/assets/tlc/downloads/pdf/industry-notices/industry_notice_22_02_english.pdf) documents a meter fare increase effective **19 December 2022**, near the end of training. Most training records precede that change; validation and test records follow it.

This is a plausible contributor to underestimation visible in the plots, not a measured causal explanation. The current feature set has no explicit pricing-regime variable. Rolling retraining or a known pricing-regime feature would require a new experiment, rather than adjusting these published scores.
''')
md('''## Measured runtime and limits

These are measured local results, not a cluster benchmark. Download time and this presentation notebook are excluded from the processing timer. The experiment uses modest fixed model settings, one seed and one temporal holdout. It does not quantify uncertainty across repeated samples.

The prediction uses realized trip measurements. Positive-fare filtering defines the analysis population. Training-only input caps can suppress genuine extreme behavior, so the residual tables and full-range plots should accompany performance claims.
''')
code('''table([{"fleet": fleet, **{stage: round(seconds / 60, 2) for stage, seconds in run["stage_seconds"].items()}}
       for fleet, run in fleets.items()])
print("Stage values above are minutes.")
print("Total processing minutes:", round(experiment["elapsed_seconds"] / 60, 2))
print(json.dumps(experiment["environment"], indent=2))
''')
for i,cell in enumerate(cells): cell['id']=f'five-year-{i:02d}'
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
Path('notebooks/nyc_taxi_portfolio.ipynb').write_text(json.dumps(notebook,indent=1))
