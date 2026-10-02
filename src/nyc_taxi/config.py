"""The fixed study scope and assumptions, kept in one place."""
from dataclasses import dataclass, asdict

YEARS = tuple(range(2020, 2025))
MONTHS = tuple(range(1, 13))
FLEETS = ('yellow', 'green')
TRAIN_BEFORE = '2023-01-01'
TEST_FROM = '2024-01-01'
FARE_CHANGE_DATE = '2022-12-19'
SAMPLE_FRACTIONS = {'yellow': 0.002, 'green': 0.05}
SEED = 42

@dataclass(frozen=True)
class CleaningRules:
    first_year: int = 2020
    last_year: int = 2024
    max_distance_miles: float = 100.0
    min_duration_minutes: float = 1.0
    max_duration_minutes: float = 360.0
    max_average_speed_mph: float = 80.0
    max_fare_dollars: float = 500.0

RULES = CleaningRules()
