from datetime import datetime
from zoneinfo import ZoneInfo
import sqlite3
import pytest
from pyspark.sql import SparkSession
from nyc_taxi.pipeline import clean, features, normalize, save_sqlite

@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2").config("spark.sql.session.timeZone", "America/New_York").getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_cleaning_boundaries_and_sqlite(spark, tmp_path):
    pickup, dropoff = datetime(2024, 1, 6, 12, tzinfo=ZoneInfo("America/New_York")), datetime(2024, 1, 6, 12, 30, tzinfo=ZoneInfo("America/New_York"))
    rows = [(pickup, dropoff, 1.0, 12.0), (pickup, dropoff, 0.0, 12.0),
            (pickup, pickup, 2.0, 12.0), (pickup, dropoff, float("nan"), 12.0),
            (None, dropoff, 2.0, 12.0), (pickup, dropoff, 2.0, -1.0)]
    frame = spark.createDataFrame(rows, "pickup_datetime timestamp, dropoff_datetime timestamp, trip_distance double, fare_amount double")
    accepted, counts, cached = clean(frame)
    assert sum(counts.values()) == 6
    assert counts["accepted"] == 1
    row = features(accepted).first()
    assert row.distance_bin == "1-2"
    assert row.pickup_weekday == 7
    assert row.pickup_hour == 12
    assert row.duration_minutes == 30
    path = tmp_path / "trips.db"
    assert save_sqlite(accepted, path, batch_size=1) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT pickup_datetime FROM trips").fetchone()[0] == "2024-01-06 12:00:00"
    assert save_sqlite(accepted.limit(0), path) == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 0
    cached.unpersist()


def test_schema_normalization(spark):
    row = ("2024-01-01 12:00:00", "2024-01-01 12:30:00", "1.5", "12", None, "1", "2", "1")
    df = spark.createDataFrame([row], "lpep_pickup_datetime string, lpep_dropoff_datetime string, trip_distance string, fare_amount string, passenger_count string, PULocationID string, DOLocationID string, payment_type string")
    result = normalize(df, "green").first()
    assert result.trip_distance == 1.5
    assert result.passenger_count is None
    with pytest.raises(ValueError, match="missing required column"):
        normalize(df.drop("payment_type"), "green")


def test_failed_export_preserves_database(spark, tmp_path, monkeypatch):
    from pyspark.sql import DataFrame
    frame = spark.createDataFrame([(datetime(2024, 1, 1), 1.0)], 'pickup_datetime timestamp, fare_amount double')
    path = tmp_path / 'trips.db'
    save_sqlite(frame, path)
    def broken_iterator(self):
        raise RuntimeError('simulated export failure')
    monkeypatch.setattr(DataFrame, 'toLocalIterator', broken_iterator)
    with pytest.raises(RuntimeError, match='simulated export failure'):
        save_sqlite(frame, path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT fare_amount FROM trips').fetchone()[0] == 1.0
    assert list(tmp_path.glob('*.db')) == [path]


def test_monthly_csv_roundtrip_and_schema_drift(spark, tmp_path):
    from nyc_taxi.pipeline import load
    from pyspark.sql import functions as F
    row = ('2024-01-01 12:00:00', '2024-01-01 12:30:00', '1.5', '12', None, '1', '2', '1')
    frame = spark.createDataFrame([row], 'tpep_pickup_datetime string, tpep_dropoff_datetime string, trip_distance string, fare_amount string, passenger_count string, PULocationID string, DOLocationID string, payment_type string')
    raw = tmp_path / 'raw'
    frame.write.parquet(str(raw / 'yellow_tripdata_2024-01.parquet'))
    frame.select(*reversed(frame.columns)).withColumn('new_optional_field', F.lit(7)).write.parquet(str(raw / 'yellow_tripdata_2024-02.parquet'))
    result = load(spark, raw, 'yellow', tmp_path / 'csv')
    assert result.count() == 2
    assert {r.trip_distance for r in result.collect()} == {1.5}
    assert result.schema['pickup_datetime'].dataType.simpleString() == 'timestamp'
    assert 'new_optional_field' not in result.columns


def test_feature_caps_and_categories_learn_only_from_training(spark):
    from nyc_taxi.pipeline import fit_preprocessing
    from nyc_taxi.cli import demo_data
    from pyspark.sql import functions as F
    training = features(demo_data(spark, 'yellow'))
    preprocessing, caps = fit_preprocessing(training)
    validation = training.limit(1).withColumn('trip_distance', F.lit(1000000.0)).withColumn('duration_minutes', F.lit(1000000.0)).withColumn('PULocationID', F.lit(999))
    transformed = preprocessing.transform(validation).first()
    assert transformed.distance_input == caps['trip_distance']
    assert transformed.duration_input == caps['duration_minutes']
    assert transformed.trip_distance == 1000000.0  # original measurements remain auditable
    assert all(__import__('math').isfinite(v) for v in transformed.features)
    assert caps['trip_distance'] < 30


def test_requested_source_period_must_be_complete(spark, tmp_path):
    from nyc_taxi.pipeline import load
    with pytest.raises(ValueError, match='Missing 2 source files'):
        load(spark, tmp_path, 'green', years=[2024], months=[1, 2])
