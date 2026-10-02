from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
import pytest
from pyspark.sql import SparkSession, functions as F
from nyc_taxi.pipeline import clean, features, normalize, load, fit_history, add_history, fit_preprocessing, regression_metrics

@pytest.fixture(scope='module')
def spark():
    s = (SparkSession.builder.master('local[2]').config('spark.ui.enabled', 'false')
         .config('spark.sql.shuffle.partitions', '2').config('spark.sql.session.timeZone', 'America/New_York').getOrCreate())
    s.sparkContext.setLogLevel('ERROR'); yield s; s.stop()


def test_cleaning_thresholds_and_first_failure(spark):
    t = datetime(2024, 1, 1, tzinfo=ZoneInfo('America/New_York'))
    specs = [(1, 1, 1, 'accepted'), (100, 360, 500, 'accepted'), (40, 30, 20, 'accepted'),
             (0, 0, -1, 'distance_missing_or_nonpositive'), (101, 30, 20, 'distance_over_limit'),
             (1, 0.5, 20, 'duration_too_short'), (1, 361, 20, 'duration_too_long'),
             (41, 30, 20, 'speed_over_limit'), (1, 30, 0, 'fare_missing_or_nonpositive'),
             (1, 30, 501, 'fare_over_limit'), (float('nan'), 30, 20, 'distance_missing_or_nonpositive'),
             (float('inf'), 30, 20, 'distance_over_limit'), (1, 30, float('inf'), 'fare_over_limit')]
    rows = [(t, t+timedelta(minutes=m), float(d), float(f), expected) for d,m,f,expected in specs]
    rows.extend([(None,t,1.,20.,'missing_timestamp'),
                 (t.replace(year=2019), t.replace(year=2019)+timedelta(minutes=30),1.,20.,'outside_period'),
                 (t.replace(year=2025), t.replace(year=2025)+timedelta(minutes=30),1.,20.,'outside_period')])
    df = spark.createDataFrame(rows, 'pickup_datetime timestamp, dropoff_datetime timestamp, trip_distance double, fare_amount double, expected string')
    accepted, counts, tagged = clean(df)
    assert sum(counts.values()) == len(rows)
    assert counts['accepted'] == accepted.count() == 3
    assert tagged.filter('quality_reason != expected').count() == 0


def test_normalization_and_monthly_schema_drift(spark, tmp_path):
    df = spark.createDataFrame([('2024-01-01 12:00:00','2024-01-01 12:30:00','1.5','12','1','2')],
        'tpep_pickup_datetime string, tpep_dropoff_datetime string, trip_distance string, fare_amount string, PULocationID string, DOLocationID string')
    df.write.parquet(str(tmp_path/'yellow_tripdata_2024-01.parquet'))
    df.select(*reversed(df.columns)).withColumn('unused', F.lit(7)).write.parquet(str(tmp_path/'yellow_tripdata_2024-02.parquet'))
    result = load(spark, tmp_path, 'yellow', years=[2024], months=[1,2])
    assert result.count() == 2
    assert result.first().trip_distance == 1.5
    assert result.schema['pickup_datetime'].dataType.simpleString() == 'timestamp'
    with pytest.raises(ValueError, match='missing required column'): normalize(df.drop('fare_amount'), 'yellow')
    with pytest.raises(ValueError, match='Missing 1 source files'): load(spark,tmp_path,'yellow',years=[2024],months=[1,2,3])


def test_training_history_fallback_and_feature_availability(spark):
    from nyc_taxi.cli import demo_data
    training = features(clean(demo_data(spark,'yellow'))[0])
    medians, fallback = fit_history(training)
    heldout = training.limit(1).withColumn('PULocationID',F.lit(999)).withColumn('DOLocationID',F.lit(999))
    enriched = add_history(heldout, medians, fallback)
    row = enriched.first()
    assert row.unseen_pair and row.historical_distance == fallback['historical_distance']
    assert row.historical_duration == fallback['historical_duration']
    train_history = add_history(training, medians, fallback)
    for setting in ('pre_trip','post_trip'):
        p = fit_preprocessing(train_history, setting)
        original = p.transform(enriched).first().features
        changed = p.transform(enriched.withColumn('trip_distance',F.lit(99.)).withColumn('duration_minutes',F.lit(350.))).first().features
        assert all(math.isfinite(v) for v in original)
        assert (original == changed) == (setting == 'pre_trip')
    # Held-out rows were never passed to fit_history: the fitted summaries stay fixed.
    assert medians.filter('PULocationID = 999').count() == 0


def test_pricing_flag_boundary_in_new_york(spark):
    times = ['2022-12-18 23:59:59','2022-12-19 00:00:00']
    df = spark.createDataFrame([(t,1.) for t in times], 'pickup_datetime string, trip_distance double').withColumn('pickup_datetime',F.to_timestamp('pickup_datetime'))
    assert [r.new_fare_regime for r in features(df).orderBy('pickup_datetime').collect()] == [0.,1.]


def test_metrics_tolerance_boundaries(spark):
    df = spark.createDataFrame([(10.,12.),(10.,15.),(10.,16.),(20.,20.)], 'fare_amount double, prediction double')
    m = regression_metrics(df)
    assert m['within_2_pct'] == 50
    assert m['within_5_pct'] == 75
    assert m['mae'] == 3.25
    assert m['rmse'] == pytest.approx(math.sqrt(65/4))
    assert m['r2'] == pytest.approx(1-(65/4)/18.75)
