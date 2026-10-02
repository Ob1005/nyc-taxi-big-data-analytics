"""Spark transformations shared by the command line and portfolio notebook."""
import csv
import json
import math
import os
import sqlite3
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
NUMERIC = ["trip_distance", "duration_minutes", "pickup_hour", "pickup_weekday", "pickup_month"]
CATEGORICAL = ["PULocationID", "DOLocationID", "distance_bin"]


PAYMENT_LABELS = {
    0: "Flex Fare (current TLC definition)", 1: "Credit card", 2: "Cash",
    3: "No charge", 4: "Dispute", 5: "Unknown (reported code)", 6: "Voided trip",
}
PAYMENT_GROUPS = {
    0: "Fare regime", 1: "Payment method", 2: "Payment method",
    3: "Payment status", 4: "Payment status", 5: "Unknown", 6: "Payment status",
}


def payment_description(code):
    """Preserve reported unknowns, absent information and unrecognized codes separately."""
    if code is None or code == "":
        return "Missing payment information", "Missing"
    try:
        numeric = int(code)
    except (ValueError, TypeError):
        return f"Unrecognized code: {code}", "Unrecognized"
    return PAYMENT_LABELS.get(numeric, f"Unrecognized code: {code}"), PAYMENT_GROUPS.get(numeric, "Unrecognized")


def payment_counts(df):
    counts = df.groupBy("payment_type").count()
    labels = F.create_map(*[expression for code, label in PAYMENT_LABELS.items()
                            for expression in (F.lit(code), F.lit(label))])
    groups = F.create_map(*[expression for code, group in PAYMENT_GROUPS.items()
                            for expression in (F.lit(code), F.lit(group))])
    code = F.col("payment_type")
    return (counts.withColumn("payment_label", F.when(code.isNull(), "Missing payment information")
                              .otherwise(F.coalesce(labels[code], F.concat(F.lit("Unrecognized code: "), code.cast("string")))))
            .withColumn("category_group", F.when(code.isNull(), "Missing")
                        .otherwise(F.coalesce(groups[code], F.lit("Unrecognized"))))
            .orderBy(F.desc("count")))


def download(root, years, months, fleets=("yellow", "green"), workers=4):
    """Validate Parquet footers, retry failures and publish downloads atomically."""
    import pyarrow.parquet as pq
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = [f"{fleet}_tripdata_{year}-{month:02d}.parquet"
             for fleet in fleets for year in years for month in months]

    def fetch(name):
        destination = root / name
        if destination.exists():
            metadata = pq.read_metadata(destination)
            return {"file": name, "status": "cached", "bytes": destination.stat().st_size, "rows": metadata.num_rows}
        for attempt in range(3):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                    temporary = Path(stream.name)
                    with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            stream.write(block)
                metadata = pq.read_metadata(temporary)
                os.replace(temporary, destination)
                return {"file": name, "status": "downloaded", "bytes": destination.stat().st_size,
                        "rows": metadata.num_rows}
            except Exception:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest = list(pool.map(fetch, names))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def normalize(df, fleet):
    prefix = "tpep" if fleet == "yellow" else "lpep"
    types = {"pickup_datetime": "timestamp", "dropoff_datetime": "timestamp",
             "trip_distance": "double", "fare_amount": "double", "passenger_count": "double",
             "PULocationID": "int", "DOLocationID": "int", "payment_type": "int"}
    actual = {c.lower(): c for c in df.columns}
    expressions = []
    for name, dtype in types.items():
        source = f"{prefix}_{name}" if name.endswith("datetime") else name
        if source.lower() not in actual:
            raise ValueError(f"{fleet}: missing required column {source}")
        expressions.append(F.col(actual[source.lower()]).cast(dtype).alias(name))
    return df.select(*expressions)


def load(spark, raw, fleet, csv_root=None, years=None, months=None):
    files = sorted(Path(raw).glob(f"{fleet}_tripdata_*.parquet"))
    if years is not None and months is not None:
        expected = {f"{fleet}_tripdata_{year}-{month:02d}.parquet" for year in years for month in months}
        missing = expected - {p.name for p in files}
        if missing:
            raise ValueError(f"Missing {len(missing)} source files; run download for the requested period")
        files = [p for p in files if p.name in expected]
    if not files:
        raise ValueError(f"No {fleet} Parquet files in {raw}; download first")
    frames = []
    for path in files:
        frame = spark.read.parquet(str(path))
        if csv_root is not None:
            target = Path(csv_root) / path.stem
            # Preserve distributed part files rather than force a single executor.
            frame.write.mode("overwrite").option("header", True).csv(str(target))
            frame = spark.read.option("header", True).csv(str(target))
        frames.append(normalize(frame, fleet))
    return reduce(lambda a, b: a.unionByName(b), frames)


def clean(df, start_year=2020, end_year=2024, persist=True):
    """Assign one first-failure reason per row; report excludes double counting."""
    duration = (F.col("dropoff_datetime").cast("long") - F.col("pickup_datetime").cast("long")) / 60
    reason = (F.when(F.col("pickup_datetime").isNull() | F.col("dropoff_datetime").isNull(), "missing_timestamp")
              .when(~F.year("pickup_datetime").between(start_year, end_year), "outside_period")
              .when(F.col("trip_distance").isNull() | F.isnan("trip_distance") |
                    (F.abs(F.col("trip_distance")) == float("inf")) | (F.col("trip_distance") <= 0), "invalid_distance")
              .when(F.col("fare_amount").isNull() | F.isnan("fare_amount") |
                    (F.abs(F.col("fare_amount")) == float("inf")) | (F.col("fare_amount") <= 0), "invalid_fare")
              .when(duration <= 0, "invalid_duration").otherwise("accepted"))
    tagged = df.withColumn("quality_reason", reason)
    if persist:
        tagged = tagged.cache()
    counts = {r.quality_reason: r["count"] for r in tagged.groupBy("quality_reason").count().collect()}
    accepted = tagged.filter(F.col("quality_reason") == "accepted").drop("quality_reason")
    return accepted, counts, tagged


def features(df):
    return (df.withColumn("duration_minutes", (F.col("dropoff_datetime").cast("long") - F.col("pickup_datetime").cast("long")) / 60)
            .withColumn("pickup_hour", F.hour("pickup_datetime"))
            .withColumn("pickup_weekday", F.dayofweek("pickup_datetime"))
            .withColumn("pickup_month", F.month("pickup_datetime"))
            .withColumn("distance_bin", F.when(F.col("trip_distance") < 1, "0-1")
                        .when(F.col("trip_distance") < 2, "1-2").when(F.col("trip_distance") < 5, "2-5")
                        .when(F.col("trip_distance") < 10, "5-10").when(F.col("trip_distance") < 20, "10-20")
                        .otherwise("20+")))


def save_sqlite(df, path, batch_size=10000):
    """Bound driver memory and replace the database only after verified completion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".db")
    os.close(handle)
    rows_written = 0
    try:
        with sqlite3.connect(temporary) as connection:
            columns = df.columns
            declaration = ", ".join(f'"{field.name}" ' + ("REAL" if field.dataType.simpleString() in ("double", "float") else "INTEGER" if field.dataType.simpleString() in ("int", "bigint") else "TEXT") for field in df.schema.fields)
            connection.execute(f"CREATE TABLE trips ({declaration})")
            insert = f'INSERT INTO trips VALUES ({",".join("?" for _ in columns)})'
            batch = []
            # toLocalIterator materializes an entire Spark partition on the JVM.
            # Insert batching alone does not bound that memory; reduce partition sizes too.
            partition_count = max(1, math.ceil(df.count() / 50000))
            # Format timestamps in Spark's configured NYC timezone, not the host timezone.
            export = df.select(*[F.date_format(F.col(field.name), "yyyy-MM-dd HH:mm:ss").alias(field.name)
                                 if field.dataType.simpleString() == "timestamp" else F.col(field.name)
                                 for field in df.schema.fields])
            for row in export.repartition(partition_count).toLocalIterator():
                batch.append(tuple(v.isoformat(sep=" ") if hasattr(v, "isoformat") else v for v in row))
                if len(batch) >= batch_size:
                    connection.executemany(insert, batch)
                    rows_written += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(insert, batch)
                rows_written += len(batch)
            connection.execute("CREATE INDEX pickup_time ON trips(pickup_datetime)")
            count = connection.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
            if count != rows_written or connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite verification failed")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return rows_written


def write_csv(df, path):
    """Collect only bounded aggregate or sampled outputs."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(df.columns)
        exported = df.select(*[
            F.date_format(F.col(field.name), "yyyy-MM-dd HH:mm:ss").alias(field.name)
            if field.dataType.simpleString() == "timestamp" else F.col(field.name)
            for field in df.schema.fields
        ])
        writer.writerows(tuple(row) for row in exported.collect())


def eda(spark, df, output, min_dropoff_trips=500):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df.createOrReplaceTempView("trips")
    analyses = {
        "top_pickups": spark.sql("SELECT PULocationID, COUNT(*) trip_count FROM trips GROUP BY PULocationID ORDER BY trip_count DESC, PULocationID LIMIT 10"),
        "hourly_trips": df.groupBy("pickup_hour").count().orderBy("pickup_hour"),
        "fare_by_distance": df.groupBy("distance_bin").agg(F.count("*").alias("trip_count"), F.avg("fare_amount").alias("mean_fare")).orderBy("distance_bin"),
        "monthly_trips": df.groupBy(F.date_format("pickup_datetime", "yyyy-MM").alias("month")).count().orderBy("month"),
        "highest_fare_dropoffs": spark.sql(f"SELECT DOLocationID, COUNT(*) trip_count, AVG(fare_amount) mean_fare FROM trips WHERE DOLocationID BETWEEN 1 AND 265 GROUP BY DOLocationID HAVING COUNT(*) >= {int(min_dropoff_trips)} ORDER BY mean_fare DESC, DOLocationID LIMIT 5"),
        "weekend_weekday": df.withColumn("day_type", F.when(F.col("pickup_weekday").isin(1, 7), "Weekend").otherwise("Weekday")).groupBy("day_type").agg(F.count("*").alias("trip_count"), F.countDistinct(F.to_date("pickup_datetime")).alias("observed_days")).withColumn("trips_per_observed_day", F.col("trip_count") / F.col("observed_days")),
        "long_trip_fares": df.filter(F.col("trip_distance") > 10).agg(F.count("*").alias("trip_count"), F.avg("fare_amount").alias("mean_fare")),
        "payment_methods": payment_counts(df),
        "fare_distribution": df.select("fare_amount").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")}
    for name, frame in analyses.items():
        write_csv(frame, output / f"{name}.csv")
    plot_demand(output)


def plot_demand(output):
    """Plot exported aggregates, including any out-of-month source records."""
    output = Path(output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    with (output / "monthly_trips.csv").open() as stream:
        monthly = list(csv.DictReader(stream))
    with (output / "hourly_trips.csv").open() as stream:
        hourly = list(csv.DictReader(stream))
    positions = list(range(len(monthly)))
    axes[0].bar(positions, [int(r["count"]) for r in monthly], color="#3977a1")
    stride = max(1, math.ceil(len(monthly) / 12))
    ticks = positions[::stride]
    axes[0].set_xticks(ticks, [monthly[i]["month"] for i in ticks], rotation=60)
    axes[0].set_title("Accepted pickups by observed month")
    axes[1].plot([int(r["pickup_hour"]) for r in hourly], [int(r["count"]) for r in hourly])
    axes[1].set_title("Pickup hour (New York local time)")
    for ax in axes:
        ax.set_ylabel("Accepted trips")
    fig.tight_layout()
    fig.savefig(output / "demand.png", dpi=150)
    plt.close(fig)


def regression_metrics(predictions):
    values = {
        name: RegressionEvaluator(labelCol="fare_amount", metricName=name).evaluate(predictions)
        for name in ("rmse", "mae", "r2")
    }
    return {name: value if math.isfinite(value) else None for name, value in values.items()}


def fit_preprocessing(training, cap_inputs=True):
    """Learn input caps and category vocabularies from training records only."""
    stages = []
    caps = {}
    if cap_inputs:
        quantiles = training.approxQuantile(["trip_distance", "duration_minutes"], [0.999], 0.0001)
        caps = dict(zip(("trip_distance", "duration_minutes"), [q[0] for q in quantiles]))
        # Preserve the raw fields for diagnostics; only the feature vector uses capped inputs.
        statement = ("SELECT *, least(trip_distance, " + str(caps["trip_distance"]) + ") AS distance_input, "
                     "least(duration_minutes, " + str(caps["duration_minutes"]) + ") AS duration_input FROM __THIS__")
        stages.append(SQLTransformer(statement=statement))
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep") for c in CATEGORICAL]
    encoder = OneHotEncoder(
        inputCols=[c + "_index" for c in CATEGORICAL],
        outputCols=[c + "_ohe" for c in CATEGORICAL], handleInvalid="keep")
    numeric = ["distance_input", "duration_input"] + NUMERIC[2:] if cap_inputs else NUMERIC
    assembler = VectorAssembler(inputCols=numeric + [c + "_ohe" for c in CATEGORICAL], outputCol="features")
    return Pipeline(stages=stages + indexers + [encoder, assembler]).fit(training), caps


def residual_diagnostics(predictions, output, name):
    """Describe extreme residuals without modifying the evaluation population."""
    frame = predictions.withColumn("error", F.col("prediction") - F.col("fare_amount"))
    frame = frame.withColumn("absolute_error", F.abs("error")).withColumn("squared_error", F.col("error") ** 2)
    totals = frame.agg(F.sum("squared_error").alias("sum_squared_error"), F.count("*").alias("rows")).first()
    worst = frame.orderBy(F.desc("absolute_error")).limit(20)
    columns = ["pickup_datetime", "trip_distance", "duration_minutes", "fare_amount", "prediction", "absolute_error"]
    write_csv(worst.select(*columns), Path(output) / f"{name}_worst_errors.csv")
    worst_total = worst.agg(F.sum("squared_error")).first()[0] or 0
    quantiles = frame.approxQuantile("absolute_error", [0.5, 0.9, 0.99], 0.001)
    by_fare = (frame.withColumn("fare_band", F.when(F.col("fare_amount") < 20, "under_20")
                               .when(F.col("fare_amount") < 50, "20_to_50").otherwise("50_plus"))
               .groupBy("fare_band").agg(F.count("*").alias("rows"), F.avg("absolute_error").alias("mae"),
                                          F.sqrt(F.avg("squared_error")).alias("rmse")))
    write_csv(by_fare.orderBy("fare_band"), Path(output) / f"{name}_error_by_fare.csv")
    return {"absolute_error_quantiles": dict(zip(("median", "p90", "p99"), quantiles)),
            "worst_20_share_of_squared_error": worst_total / totals.sum_squared_error if totals.sum_squared_error else 0}


def plot_predictions(predictions, path, title, seed=42, max_points=2000):
    """Use a seeded hash ranking, so previews do not favor the first file or partition."""
    # A distributed top-k selection collects only the bounded preview.
    ranking = F.xxhash64("pickup_datetime", "dropoff_datetime", "trip_distance", "fare_amount",
                        "PULocationID", "DOLocationID", F.lit(seed))
    points = predictions.orderBy(ranking).limit(max_points).select("fare_amount", "prediction").collect()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    actual, predicted = zip(*[(row.fare_amount, row.prediction) for row in points])
    for ax in axes:
        ax.scatter(actual, predicted, alpha=0.3, s=8)
        ax.set(xlabel="Actual metered fare ($)", ylabel="Predicted metered fare ($)")
    lower, upper = min(0, min(predicted)), max(max(actual), max(predicted))
    axes[0].plot([lower, upper], [lower, upper], "r--")
    axes[0].set_title("Full preview range")
    axes[1].plot([0, 100], [0, 100], "r--")
    axes[1].set(xlim=(0, 100), ylim=(0, 100), title="Typical fares: $0–100 (zoom)")
    fig.suptitle(title + f" — seeded preview, n={len(points):,}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(df, output, train_before, validation_before, fraction=1.0, seed=42, compare_sample_size=True):
    """Compare sample sizes on fixed validation data and reserve test data for the final winner."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    larger_fraction = min(1.0, fraction * 2) if compare_sample_size else fraction
    # The larger pool is sampled once. Validation/test retain the starting fraction;
    # only training grows for the sample-size comparison.
    pool = df.sample(False, larger_fraction, seed).cache()
    large_training = pool.filter(F.col("pickup_datetime") < train_before).cache()
    ratio = fraction / larger_fraction
    training = large_training.sample(False, ratio, seed + 1).cache()
    validation = pool.filter((F.col("pickup_datetime") >= train_before) &
                             (F.col("pickup_datetime") < validation_before)).sample(False, ratio, seed + 2).cache()
    test = pool.filter(F.col("pickup_datetime") >= validation_before).sample(False, ratio, seed + 3).cache()
    cached = [pool, large_training, training, validation, test]
    try:
        counts = [frame.count() for frame in (training, validation, test)]
        large_count = large_training.count()
        if min(counts) < 20:
            raise ValueError(f"Need at least 20 rows per temporal split; got {counts}.")
        preprocessing, caps = fit_preprocessing(training)
        train_vec = preprocessing.transform(training).cache()
        valid_vec = preprocessing.transform(validation).cache()
        cached.extend([train_vec, valid_vec])
        estimators = [LinearRegression(labelCol="fare_amount", regParam=0.1),
                      RandomForestRegressor(labelCol="fare_amount", numTrees=20, maxDepth=5, seed=seed),
                      GBTRegressor(labelCol="fare_amount", maxIter=20, maxDepth=4, seed=seed)]
        results, fitted = [], []
        diagnostics = {}
        for estimator in estimators:
            model_started = time.perf_counter()
            model = estimator.fit(train_vec)
            predictions = model.transform(valid_vec).cache()
            try:
                score = regression_metrics(predictions)
                name = type(estimator).__name__
                diagnostics[name] = residual_diagnostics(predictions, output, "validation_" + name)
                results.append({"model": name, **score, "fit_and_evaluation_seconds": time.perf_counter() - model_started})
            finally:
                predictions.unpersist()
            fitted.append(model)
        winner = min(range(len(results)), key=lambda i: results[i]["rmse"])
        selected = fitted[winner]
        selected_training = training
        chosen_preprocessing = preprocessing
        sample_comparison = [{"training_rows": counts[0], "training_fraction": fraction,
                              "validation": {k: results[winner][k] for k in ("rmse", "mae", "r2")}}]
        if compare_sample_size and large_count > counts[0]:
            larger_preprocessing, larger_caps = fit_preprocessing(large_training)
            larger_model = estimators[winner].copy({}).fit(larger_preprocessing.transform(large_training))
            larger_score = regression_metrics(larger_model.transform(larger_preprocessing.transform(validation)))
            sample_comparison.append({"training_rows": large_count, "training_fraction": larger_fraction,
                                      "validation": larger_score})
            if larger_score["rmse"] < results[winner]["rmse"]:
                selected, selected_training = larger_model, large_training
                chosen_preprocessing, caps = larger_preprocessing, larger_caps
        # Investigate the previously observed extreme linear-regression residuals
        # on the same base train/validation records, without feature caps.
        uncapped_preprocessing, _ = fit_preprocessing(training, cap_inputs=False)
        uncapped_model = estimators[0].copy({}).fit(uncapped_preprocessing.transform(training))
        uncapped_predictions = uncapped_model.transform(uncapped_preprocessing.transform(validation)).cache()
        try:
            uncapped_score = regression_metrics(uncapped_predictions)
            uncapped_diagnostics = residual_diagnostics(uncapped_predictions, output, "validation_linear_uncapped")
        finally:
            uncapped_predictions.unpersist()
        predictions = selected.transform(chosen_preprocessing.transform(test)).cache()
        cached.append(predictions)
        baseline = selected_training.agg(F.avg("fare_amount")).first()[0]
        report = {
            "split_counts": dict(zip(("base_train", "validation", "test"), counts)),
            "selected_training_rows": large_count if selected_training is large_training else counts[0],
            "sample_fraction": fraction, "seed": seed,
            "train_before": train_before, "validation_before": validation_before,
            "validation": results, "selected_model": results[winner]["model"],
            "sample_size_comparison": sample_comparison,
            "training_input_caps_p999": caps,
            "uncapped_linear_validation": uncapped_score,
            "uncapped_linear_diagnostics": uncapped_diagnostics,
            "validation_diagnostics": diagnostics,
            "test": regression_metrics(predictions),
            "test_diagnostics": residual_diagnostics(predictions, output, "test_selected"),
            "mean_baseline_test": regression_metrics(test.withColumn("prediction", F.lit(baseline))),
            "prediction_context": "completed-trip metered fare, excluding tips and surcharges",
            "elapsed_seconds": time.perf_counter() - started,
        }
        chosen_preprocessing.write().overwrite().save(str(output / "preprocessing"))
        selected.write().overwrite().save(str(output / "model"))
        plot_predictions(predictions, output / "actual_vs_predicted.png", results[winner]["model"], seed)
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
    finally:
        for frame in reversed(cached):
            frame.unpersist()
