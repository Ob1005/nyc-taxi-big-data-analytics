"""Run the fixed five-year study or a small synthetic execution check."""
import argparse
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pyspark
from pyspark.sql import SparkSession, functions as F
from .config import YEARS, MONTHS, FLEETS, RULES, SAMPLE_FRACTIONS, TRAIN_BEFORE, TEST_FROM
from .pipeline import clean, download, eda, features, load, train, write_json, write_table


def demo_data(spark, fleet):
    rng = random.Random(42 if fleet == 'yellow' else 43)
    records = []
    for i in range(900):
        pickup = datetime(2024, 1, 1, tzinfo=ZoneInfo('America/New_York')) + timedelta(days=i % 90, hours=i % 24)
        distance = rng.uniform(0.5, 25)
        duration = distance * rng.uniform(2, 5)
        fare = max(1.0, 3 + 2.5 * distance + 0.3 * duration + rng.gauss(0, 2))
        records.append((pickup, pickup+timedelta(minutes=duration), distance, fare,
                        rng.choice([1, 2, 3]), rng.choice([4, 5, 6])))
    return spark.createDataFrame(records, 'pickup_datetime timestamp, dropoff_datetime timestamp, trip_distance double, fare_amount double, PULocationID int, DOLocationID int')


def main():
    parser = argparse.ArgumentParser(description='Five-year NYC taxi fare study')
    parser.add_argument('command', choices=['download', 'run', 'demo'])
    parser.add_argument('--yellow-sample-fraction', type=float, default=SAMPLE_FRACTIONS['yellow'])
    parser.add_argument('--green-sample-fraction', type=float, default=SAMPLE_FRACTIONS['green'])
    args = parser.parse_args()
    fractions = {'yellow': args.yellow_sample_fraction, 'green': args.green_sample_fraction}
    if any(not 0 < v <= 1 for v in fractions.values()): parser.error('Sample fractions must be in (0, 1]')
    if args.command == 'download':
        download(Path('data/raw'), YEARS, MONTHS, FLEETS)
        print('Downloaded source files; ready for nyc-taxi run.'); return
    started = time.perf_counter()
    mode = 'demo' if args.command == 'demo' else 'real'
    root = Path('outputs') / mode; root.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.master('local[2]').appName('NYC taxi fare study')
             .config('spark.sql.session.timeZone', 'America/New_York')
             .config('spark.sql.shuffle.partitions', '64' if mode == 'real' else '4')
             .config('spark.sql.files.maxPartitionBytes', str(32*1024*1024))
             .config('spark.driver.memory', '4g').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    environment = {'python': sys.version, 'spark': pyspark.__version__, 'platform': platform.platform(),
                   'java': subprocess.run(['java', '-version'], capture_output=True, text=True).stderr.strip(),
                   'master': 'local[2]', 'driver_memory': '4g', 'timezone': 'America/New_York'}
    reports = {}
    try:
        for fleet in FLEETS:
            print(f'Starting {fleet}: cleaning all source trips', flush=True)
            fleet_started = time.perf_counter(); destination = root / fleet; destination.mkdir(parents=True, exist_ok=True)
            raw = demo_data(spark, fleet) if mode == 'demo' else load(spark, 'data/raw', fleet)
            accepted, counts, tagged = clean(raw)
            # Re-read this source example on every run rather than copying an old finding.
            examples = write_table(tagged.filter(F.col('trip_distance') == 6860.8)
                .select('pickup_datetime', 'trip_distance', 'duration_minutes', 'fare_amount', 'quality_reason').limit(1),
                destination / 'data_quality_examples.json')
            sources = []
            if mode == 'real':
                import pyarrow.parquet as pq
                sources = [{'file': f'{fleet}_tripdata_{y}-{m:02d}.parquet',
                            'rows': pq.read_metadata(Path('data/raw') / f'{fleet}_tripdata_{y}-{m:02d}.parquet').num_rows}
                           for y in YEARS for m in MONTHS]
                if sum(s['rows'] for s in sources) != sum(counts.values()):
                    raise RuntimeError('Source and cleaning row totals differ')
            quality = {'source_rows': sum(counts.values()), 'accepted_rows': counts.get('accepted', 0),
                       'first_failure_counts': counts, 'thresholds': asdict(RULES), 'examples': examples}
            write_json(quality, destination / 'quality.json')
            cleaned_path = Path('data/cleaned') / mode / fleet
            accepted.write.mode('overwrite').parquet(str(cleaned_path))
            enriched = features(spark.read.parquet(str(cleaned_path)))
            cleaning_seconds = time.perf_counter()-fleet_started
            print(f'{fleet}: {quality["accepted_rows"]:,} accepted; starting analysis', flush=True)
            stage = time.perf_counter(); eda(spark, enriched, destination / 'eda', minimum=1 if mode=='demo' else 500)
            eda_seconds = time.perf_counter()-stage
            stage = time.perf_counter()
            report = train(enriched, destination / 'modeling',
                           '2024-02-01' if mode=='demo' else TRAIN_BEFORE,
                           '2024-03-01' if mode=='demo' else TEST_FROM,
                           1.0 if mode=='demo' else fractions[fleet])
            reports[fleet] = {'synthetic': mode=='demo', 'source_files': sources, 'quality': quality, 'modeling': report,
                             'stage_seconds': {'cleaning': cleaning_seconds, 'eda': eda_seconds, 'modeling': time.perf_counter()-stage},
                             'elapsed_seconds': time.perf_counter()-fleet_started}
            write_json(reports[fleet], destination / 'run.json')
    finally:
        spark.stop()
    experiment = {'fleets': reports, 'environment': environment, 'elapsed_seconds': time.perf_counter()-started}
    write_json(experiment, root / 'experiment.json')
    if mode == 'real':
        from .reporting import publish
        publish(root)
    print(f'Finished {mode} run in {experiment["elapsed_seconds"] / 60:.2f} minutes; results in {root}', flush=True)

if __name__ == '__main__': main()
