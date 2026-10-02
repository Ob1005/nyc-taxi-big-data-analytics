"""The cleaning, Spark SQL analysis and fare modelling workflow."""
import json
import math
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from .config import YEARS, MONTHS, RULES, FARE_CHANGE_DATE, SEED

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
CATEGORICAL = ['PULocationID', 'DOLocationID']
PRE_TRIP = ['pickup_hour', 'pickup_weekday', 'pickup_month', 'new_fare_regime',
            'historical_distance', 'historical_duration']
FEATURE_SETS = {'pre_trip': PRE_TRIP, 'post_trip': PRE_TRIP + ['trip_distance', 'duration_minutes']}

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
    prefix = 'tpep' if fleet == 'yellow' else 'lpep'
    types = {'pickup_datetime': 'timestamp', 'dropoff_datetime': 'timestamp',
             'trip_distance': 'double', 'fare_amount': 'double',
             'PULocationID': 'int', 'DOLocationID': 'int'}
    actual = {c.lower(): c for c in df.columns}
    columns = []
    for name, dtype in types.items():
        source = f'{prefix}_{name}' if name.endswith('datetime') else name
        if source.lower() not in actual:
            raise ValueError(f'{fleet}: missing required column {source}')
        columns.append(F.col(actual[source.lower()]).cast(dtype).alias(name))
    return df.select(*columns)


def load(spark, raw, fleet, years=YEARS, months=MONTHS):
    paths = [Path(raw) / f'{fleet}_tripdata_{year}-{month:02d}.parquet'
             for year in years for month in months]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise ValueError(f'Missing {len(missing)} source files; run nyc-taxi download first')
    return reduce(lambda a, b: a.unionByName(b),
                  [normalize(spark.read.parquet(str(p)), fleet) for p in paths])


def cleaning_reason(df):
    """First failing rule wins; order is also the rejection report order."""
    distance, fare = F.col('trip_distance'), F.col('fare_amount')
    duration = F.col('duration_minutes')
    return (F.when(F.col('pickup_datetime').isNull() | F.col('dropoff_datetime').isNull(), 'missing_timestamp')
            .when(~F.year('pickup_datetime').between(RULES.first_year, RULES.last_year), 'outside_period')
            .when(distance.isNull() | F.isnan(distance) | (distance <= 0), 'distance_missing_or_nonpositive')
            .when(distance > RULES.max_distance_miles, 'distance_over_limit')
            .when(duration < RULES.min_duration_minutes, 'duration_too_short')
            .when(duration > RULES.max_duration_minutes, 'duration_too_long')
            .when(distance * 60 / duration > RULES.max_average_speed_mph, 'speed_over_limit')
            .when(fare.isNull() | F.isnan(fare) | (fare <= 0), 'fare_missing_or_nonpositive')
            .when(fare > RULES.max_fare_dollars, 'fare_over_limit')
            .otherwise('accepted'))


def clean(df):
    tagged = df.withColumn('duration_minutes',
        (F.col('dropoff_datetime').cast('long') - F.col('pickup_datetime').cast('long')) / 60)
    tagged = tagged.withColumn('quality_reason', cleaning_reason(tagged))
    counts = {r.quality_reason: r['count'] for r in tagged.groupBy('quality_reason').count().collect()}
    accepted = tagged.filter(F.col('quality_reason') == 'accepted').drop('quality_reason')
    return accepted, counts, tagged


def features(df):
    return (df.withColumn('pickup_hour', F.hour('pickup_datetime'))
            .withColumn('pickup_weekday', F.dayofweek('pickup_datetime'))
            .withColumn('pickup_month', F.month('pickup_datetime'))
            .withColumn('new_fare_regime', (F.col('pickup_datetime') >= F.lit(FARE_CHANGE_DATE).cast('timestamp')).cast('double'))
            .withColumn('distance_band', F.when(F.col('trip_distance') < 1, '0–1')
                        .when(F.col('trip_distance') < 2, '1–2').when(F.col('trip_distance') < 5, '2–5')
                        .when(F.col('trip_distance') < 10, '5–10').when(F.col('trip_distance') < 20, '10–20')
                        .otherwise('20–100')))


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def write_table(df, path):
    # Only small aggregates and bounded residual previews reach the driver.
    columns = [F.date_format(f.name, 'yyyy-MM-dd HH:mm:ss').alias(f.name)
               if f.dataType.simpleString() == 'timestamp' else F.col(f.name) for f in df.schema.fields]
    rows = [r.asDict() for r in df.select(*columns).collect()]
    write_json(rows, path)
    return rows


EDA_QUERIES = {
    'monthly_trips': "SELECT date_format(pickup_datetime, 'yyyy-MM') month, COUNT(*) trip_count FROM trips GROUP BY 1 ORDER BY 1",
    'hourly_trips': 'SELECT pickup_hour, COUNT(*) trip_count FROM trips GROUP BY pickup_hour ORDER BY pickup_hour',
    'fare_by_distance': 'SELECT distance_band, COUNT(*) trip_count, AVG(fare_amount) mean_fare FROM trips GROUP BY distance_band ORDER BY MIN(trip_distance)',
    'highest_fare_dropoffs': 'SELECT DOLocationID, COUNT(*) trip_count, AVG(fare_amount) mean_fare FROM trips WHERE DOLocationID BETWEEN 1 AND 263 GROUP BY DOLocationID HAVING COUNT(*) >= {minimum} ORDER BY mean_fare DESC, DOLocationID LIMIT 5',
}


def eda(spark, df, output, minimum=500):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    df.createOrReplaceTempView('trips')
    tables = {name: write_table(spark.sql(query.format(minimum=minimum)), output / f'{name}.json')
              for name, query in EDA_QUERIES.items()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs = [
        ('monthly_trips', 'month', 'trip_count', 'Accepted pickups by month', 'Trips'),
        ('hourly_trips', 'pickup_hour', 'trip_count', 'Pickup hour (New York time)', 'Trips'),
        ('fare_by_distance', 'distance_band', 'mean_fare', 'Mean fare by distance band (miles)', 'Fare ($)'),
        ('highest_fare_dropoffs', 'DOLocationID', 'mean_fare', 'Highest mean-fare destination zones', 'Fare ($)'),
    ]
    for name, x, y, title, ylabel in specs:
        rows = tables[name]; fig, ax = plt.subplots(figsize=(9, 3.5))
        positions = list(range(len(rows)))
        ax.bar(positions, [r[y] for r in rows], color='#3977a1')
        stride = max(1, math.ceil(len(rows)/12))
        ax.set_xticks(positions[::stride], [str(rows[i][x]) for i in positions[::stride]], rotation=45)
        ax.set(title=title, ylabel=ylabel)
        fig.tight_layout(); fig.savefig(output / f'{name}.png', dpi=140); plt.close(fig)
    return tables


def fit_history(training):
    """Route summaries and fallback come exclusively from the training sample."""
    medians = training.groupBy(*CATEGORICAL).agg(
        F.percentile_approx('trip_distance', 0.5, 10000).alias('historical_distance'),
        F.percentile_approx('duration_minutes', 0.5, 10000).alias('historical_duration'))
    fallback = training.agg(F.percentile_approx('trip_distance', 0.5, 10000),
                            F.percentile_approx('duration_minutes', 0.5, 10000)).first()
    return medians, {'historical_distance': fallback[0], 'historical_duration': fallback[1]}


def add_history(df, medians, fallback):
    result = df.join(F.broadcast(medians), CATEGORICAL, 'left')
    return (result.withColumn('unseen_pair', F.col('historical_distance').isNull())
            .fillna(fallback))


def fit_preprocessing(training, feature_set):
    indexers = [StringIndexer(inputCol=c, outputCol=c+'_index', handleInvalid='keep') for c in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[c+'_index' for c in CATEGORICAL],
                            outputCols=[c+'_ohe' for c in CATEGORICAL], handleInvalid='keep')
    assembler = VectorAssembler(inputCols=FEATURE_SETS[feature_set] + [c+'_ohe' for c in CATEGORICAL], outputCol='features')
    return Pipeline(stages=indexers + [encoder, assembler]).fit(training)


def regression_metrics(predictions):
    # One aggregation avoids repeatedly scanning predictions for each metric.
    row = predictions.agg(F.count('*').alias('n'), F.avg('fare_amount').alias('mean'),
        F.var_pop('fare_amount').alias('variance'),
        F.avg(F.abs(F.col('prediction')-F.col('fare_amount'))).alias('mae'),
        F.avg((F.col('prediction')-F.col('fare_amount'))**2).alias('mse'),
        (100*F.avg((F.abs(F.col('prediction')-F.col('fare_amount'))<=2).cast('double'))).alias('within_2_pct'),
        (100*F.avg((F.abs(F.col('prediction')-F.col('fare_amount'))<=5).cast('double'))).alias('within_5_pct')).first()
    return {'mae': row.mae, 'rmse': math.sqrt(row.mse),
            'r2': 1-row.mse/row.variance if row.variance else None,
            'within_2_pct': row.within_2_pct, 'within_5_pct': row.within_5_pct}


def residual_diagnostics(predictions, output, name):
    frame = predictions.withColumn('absolute_error', F.abs(F.col('prediction')-F.col('fare_amount')))
    bands = frame.withColumn('fare_band', F.when(F.col('fare_amount')<20, 'under_20')
                             .when(F.col('fare_amount')<50, '20_to_50').otherwise('50_plus'))
    write_table(bands.groupBy('fare_band').agg(F.count('*').alias('rows'), F.avg('absolute_error').alias('mae'),
        F.sqrt(F.avg(F.col('absolute_error')**2)).alias('rmse')).orderBy('fare_band'), output / f'{name}_error_by_fare.json')
    write_table(frame.orderBy(F.desc('absolute_error'), 'pickup_datetime', 'PULocationID', 'DOLocationID').limit(20)
        .select('pickup_datetime', *CATEGORICAL, 'trip_distance', 'duration_minutes', 'fare_amount', 'prediction', 'absolute_error'),
        output / f'{name}_worst_errors.json')


def train(df, output, train_before, test_from, fraction):
    """Select each setting on validation, then evaluate its winner once on test."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pool = df.sample(False, fraction, SEED).cache()
    splits = [pool.filter(F.col('pickup_datetime') < train_before),
              pool.filter((F.col('pickup_datetime') >= train_before) & (F.col('pickup_datetime') < test_from)),
              pool.filter(F.col('pickup_datetime') >= test_from)]
    training, validation, test = splits
    try:
        counts = dict(zip(('train', 'validation', 'test'), [s.count() for s in splits]))
        if min(counts.values()) < 20:
            raise ValueError(f'Need at least 20 rows in each split: {counts}')
        medians, fallback = fit_history(training)
        medians = medians.cache(); medians.count()
        medians.write.mode('overwrite').parquet(str(output / 'zone_pair_medians'))
        write_json(fallback, output / 'fallback.json')
        history = [add_history(s, medians, fallback).cache() for s in splits]
        baseline_mean = training.agg(F.avg('fare_amount')).first()[0]
        report = {'sample_fraction': fraction, 'seed': SEED, 'split_counts': counts,
                  'train_before': train_before, 'test_from': test_from, 'fallback': fallback,
                  'training_pricing_regimes': [r.asDict() for r in training.groupBy('new_fare_regime').count().orderBy('new_fare_regime').collect()],
                  'test_unseen_pair_rows': history[2].filter('unseen_pair').count(),
                  'mean_baseline_test': regression_metrics(test.withColumn('prediction', F.lit(baseline_mean))),
                  'feature_sets': {}}
        for setting in FEATURE_SETS:
            preprocessing = fit_preprocessing(history[0], setting)
            train_vec, valid_vec = [preprocessing.transform(s).cache() for s in history[:2]]
            estimators = [LinearRegression(labelCol='fare_amount', regParam=0.1),
                          RandomForestRegressor(labelCol='fare_amount', numTrees=20, maxDepth=5, seed=SEED),
                          GBTRegressor(labelCol='fare_amount', maxIter=20, maxDepth=4, seed=SEED)]
            results, models = [], []
            for estimator in estimators:
                model = estimator.fit(train_vec)
                score = regression_metrics(model.transform(valid_vec))
                results.append({'model': type(estimator).__name__, **score}); models.append(model)
                print(f'{setting}: {type(estimator).__name__} validation RMSE {score["rmse"]:.3f}', flush=True)
            winner = min(range(len(results)), key=lambda i: results[i]['rmse'])
            predictions = models[winner].transform(preprocessing.transform(history[2])).cache()
            score = regression_metrics(predictions)
            residual_diagnostics(predictions, output, setting)
            folder = output / setting
            preprocessing.write().overwrite().save(str(folder / 'preprocessing'))
            models[winner].write().overwrite().save(str(folder / 'model'))
            report['feature_sets'][setting] = {'validation': results, 'selected_model': results[winner]['model'], 'test': score}
            predictions.unpersist(); train_vec.unpersist(); valid_vec.unpersist()
        write_json(report, output / 'metrics.json')
        return report
    finally:
        for frame in locals().get('history', []): frame.unpersist()
        if 'medians' in locals(): medians.unpersist()
        pool.unpersist()
