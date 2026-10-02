"""Reproducible full-data and synthetic demonstration entry points."""
import argparse
import json
import random
import platform
import shutil
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.sql import SparkSession
from .pipeline import clean, download, eda, features, load, save_sqlite, train


def demo_data(spark, fleet):
    rng = random.Random(42 if fleet == "yellow" else 43)
    records = []
    for i in range(900):
        pickup = datetime(2024, 1, 1, tzinfo=ZoneInfo("America/New_York")) + timedelta(days=i % 90, hours=i % 24)
        distance = rng.uniform(0.2, 25)
        duration = distance * rng.uniform(2, 5)
        fare = max(1.0, 3 + 2.5 * distance + 0.3 * duration + rng.gauss(0, 2))
        records.append((pickup, pickup + timedelta(minutes=duration), distance, fare, 1.0,
                        rng.choice([1, 2, 3]), rng.choice([4, 5, 6]), rng.choice([1, 2])))
    return spark.createDataFrame(records, "pickup_datetime timestamp, dropoff_datetime timestamp, trip_distance double, fare_amount double, passenger_count double, PULocationID int, DOLocationID int, payment_type int")


def main():
    parser = argparse.ArgumentParser(description="NYC taxi analytics and completed-trip fare modeling")
    parser.add_argument("command", choices=["download", "run", "demo"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2025)))
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--fleets", nargs="+", choices=["yellow", "green"], default=["yellow", "green"])
    parser.add_argument("--csv", action="store_true", help="Convert monthly Parquet to partitioned CSV and read it back")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Optional override for both fleets")
    parser.add_argument("--yellow-sample-fraction", type=float, default=0.002)
    parser.add_argument("--green-sample-fraction", type=float, default=0.05)
    parser.add_argument("--sqlite-limit", type=int, default=100000, help="Portable snapshot rows; 0 exports all accepted rows")
    parser.add_argument("--no-sample-comparison", action="store_true")
    parser.add_argument("--train-before", default="2023-01-01")
    parser.add_argument("--validation-before", default="2024-01-01")
    parser.add_argument("--master", default="local[2]")
    args = parser.parse_args()
    if any(y < 2020 or y > 2024 for y in args.years) or any(m < 1 or m > 12 for m in args.months):
        parser.error("Use years 2020–2024 and months 1–12")
    fractions = [args.yellow_sample_fraction, args.green_sample_fraction]
    if args.sample_fraction is not None:
        fractions.append(args.sample_fraction)
    if any(not 0 < fraction <= 1 for fraction in fractions):
        parser.error("sample fractions must be in (0, 1]")
    if args.sqlite_limit < 0:
        parser.error("SQLite limit must be zero or positive")
    try:
        if datetime.fromisoformat(args.train_before) >= datetime.fromisoformat(args.validation_before):
            parser.error("train boundary must precede validation boundary")
    except ValueError:
        parser.error("Use ISO date boundaries")
    if args.command == "download":
        print(json.dumps(download(args.data / "raw", args.years, args.months, args.fleets), indent=2))
        return
    started = time.perf_counter()
    import pyspark
    spark = (SparkSession.builder.master(args.master).appName("NYC Taxi Portfolio")
             .config("spark.sql.session.timeZone", "America/New_York")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.sql.files.maxPartitionBytes", str(32 * 1024 * 1024))
             .config("spark.driver.memory", "4g").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    environment = {"python": sys.version, "spark": pyspark.__version__, "platform": platform.platform(),
                   "java": subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.strip(),
                   "master": args.master, "driver_memory": "4g", "timezone": "America/New_York"}
    mode = "demo" if args.command == "demo" else "real"
    root = args.output / mode
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    try:
        for fleet in args.fleets:
            fleet_started = time.perf_counter()
            timings = {}
            destination = root / fleet
            destination.mkdir(parents=True, exist_ok=True)
            stage_started = time.perf_counter()
            raw = demo_data(spark, fleet) if mode == "demo" else load(
                spark, args.data / "raw", fleet, args.data / "csv" if args.csv else None,
                args.years, args.months)
            accepted, quality, tagged = clean(raw, min(args.years), max(args.years), persist=False)
            (destination / "quality.json").write_text(json.dumps(quality, indent=2))
            cleaned_path = args.data / "cleaned" / mode / fleet
            # Persist all accepted data, then release the long monthly input plan.
            accepted.write.mode("overwrite").parquet(str(cleaned_path))
            accepted = spark.read.parquet(str(cleaned_path))
            timings["load_clean_and_parquet_seconds"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if args.sqlite_limit == 0:
                estimate = quality.get("accepted", 0) * 130
                if estimate > shutil.disk_usage(destination).free * 0.7:
                    raise RuntimeError("Insufficient disk headroom for full SQLite export; use --sqlite-limit 100000")
            sqlite_frame = accepted.limit(args.sqlite_limit) if args.sqlite_limit else accepted
            saved = save_sqlite(sqlite_frame, destination / f"nyc_{fleet}_taxi.db")
            timings["sqlite_seconds"] = time.perf_counter() - stage_started
            # Keep full-data scans streaming; cache only the much smaller modeling pool.
            enriched = features(accepted)
            stage_started = time.perf_counter()
            eda(spark, enriched, destination / "eda", min_dropoff_trips=1 if mode == "demo" else 500)
            timings["eda_seconds"] = time.perf_counter() - stage_started
            fraction = args.sample_fraction if args.sample_fraction is not None else (
                args.yellow_sample_fraction if fleet == "yellow" else args.green_sample_fraction)
            stage_started = time.perf_counter()
            report = train(enriched, destination / "modeling",
                           "2024-02-01" if mode == "demo" else args.train_before,
                           "2024-03-01" if mode == "demo" else args.validation_before,
                           1.0 if mode == "demo" else fraction,
                           compare_sample_size=not args.no_sample_comparison and mode != "demo")
            timings["modeling_seconds"] = time.perf_counter() - stage_started
            if mode == "demo":
                sources = []
            else:
                import pyarrow.parquet as pq
                sources = [{"file": f"{fleet}_tripdata_{year}-{month:02d}.parquet",
                            "rows": pq.read_metadata(args.data / "raw" / f"{fleet}_tripdata_{year}-{month:02d}.parquet").num_rows}
                           for year in args.years for month in args.months]
            fleet_report = {"synthetic": mode == "demo", "source_files": sources,
                            "years": args.years, "months": args.months,
                            "sqlite_rows": saved, "sqlite_scope": "full" if not args.sqlite_limit else "bounded_snapshot",
                            "sqlite_limit": args.sqlite_limit, "full_cleaned_rows": quality.get("accepted", 0),
                            "quality": quality, "modeling": report, "stage_seconds": timings,
                            "elapsed_seconds": time.perf_counter() - fleet_started, "environment": environment}
            (destination / "run.json").write_text(json.dumps(fleet_report, indent=2))
            reports[fleet] = fleet_report
            print(f"{fleet}: {quality.get('accepted', 0):,} full cleaned trips; {saved:,} SQLite snapshot rows; "
                  f"{report['selected_training_rows']:,} selected training rows; outputs: {destination}", flush=True)
    finally:
        spark.stop()
    (root / "experiment.json").write_text(json.dumps({"fleets": reports, "environment": environment,
        "elapsed_seconds": time.perf_counter() - started, "csv_conversion": args.csv}, indent=2))


if __name__ == "__main__":
    main()
