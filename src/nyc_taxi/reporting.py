"""Publish compact, measured results and regenerate the portfolio documentation."""
import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from .pipeline import payment_description


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def publish(run_root, destination=Path('docs/results')):
    run_root, destination = Path(run_root), Path(destination)
    experiment = json.loads((run_root / 'experiment.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / 'experiment.json', destination / 'experiment.json')
    shutil.copy2('data/raw/manifest.json', destination / 'download-manifest.json')
    summary = ['# Five-year results: NYC yellow and green taxis, 2020–2024', '',
               'These are measured results from a local run over all 120 monthly source files. '
               'All accepted rows contribute to cleaning and EDA. Modeling uses separate seeded samples.', '',
               '## Coverage and evaluation', '',
               '| Fleet | Raw rows | Accepted rows | Base training | Selected training | Validation | Test |',
               '|---|---:|---:|---:|---:|---:|---:|']
    metrics_table = ['| Fleet | Selected model | Test RMSE ($) | Test MAE ($) | Test R² | Mean baseline RMSE ($) |',
                     '|---|---|---:|---:|---:|---:|']
    findings = []
    for fleet, run in experiment['fleets'].items():
        target = destination / fleet
        target.mkdir(exist_ok=True)
        source = run_root / fleet
        for filename in ('quality.json', 'run.json'):
            shutil.copy2(source / filename, target / filename)
        for path in (source / 'eda').iterdir():
            if path.suffix in ('.csv', '.png'):
                shutil.copy2(path, target / path.name)
        for path in (source / 'modeling').iterdir():
            if path.suffix in ('.json', '.csv', '.png'):
                shutil.copy2(path, target / path.name)
        # An explicitly descriptive preview of the SQLite snapshot, not a model sample.
        with sqlite3.connect(source / f'nyc_{fleet}_taxi.db') as connection:
            cursor = connection.execute('SELECT * FROM trips LIMIT 8')
            names = [description[0] for description in cursor.description]
            preview = [dict(zip(names, row)) for row in cursor.fetchall()]
        for row in preview:
            pickup = datetime.fromisoformat(row['pickup_datetime'])
            dropoff = datetime.fromisoformat(row['dropoff_datetime'])
            row.update(duration_minutes=(dropoff-pickup).total_seconds()/60,
                       pickup_hour=pickup.hour, pickup_weekday=(pickup.weekday()+1)%7+1,
                       pickup_month=pickup.month)
            distance = row['trip_distance']
            row['distance_bin'] = next(label for bound, label in [(1,'0-1'),(2,'1-2'),(5,'2-5'),(10,'5-10'),(20,'10-20'),(float('inf'),'20+')] if distance < bound)
        with (target / 'feature_preview.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(preview[0]))
            writer.writeheader()
            writer.writerows(preview)
        m = run['modeling']
        counts = m['split_counts']
        raw = sum(item['rows'] for item in run['source_files'])
        summary.append(f"| {fleet.title()} | {raw:,} | {run['full_cleaned_rows']:,} | {counts['base_train']:,} | "
                       f"{m['selected_training_rows']:,} | {counts['validation']:,} | {counts['test']:,} |")
        metrics_table.append(f"| {fleet.title()} | {m['selected_model']} | {m['test']['rmse']:.2f} | {m['test']['mae']:.2f} | "
                             f"{m['test']['r2']:.3f} | {m['mean_baseline_test']['rmse']:.2f} |")
        hours = rows(target / 'hourly_trips.csv')
        busiest = max(hours, key=lambda r: int(r['count']))
        monthly = rows(target / 'monthly_trips.csv')
        by_year = {}
        for row in monthly:
            year = row['month'][:4]
            by_year[year] = by_year.get(year, 0) + int(row['count'])
        total_2020, total_2024 = by_year.get('2020',0), by_year.get('2024',0)
        fare_summary = {row['summary']: row['fare_amount'] for row in rows(target / 'fare_distribution.csv')}
        payment_records = rows(target / 'payment_methods.csv')
        payment = max((row for row in payment_records if row['payment_type'] in ('1', '2')), key=lambda row: int(row['count']))
        payment_label = payment_description(payment['payment_type'])[0].lower()
        findings.extend([f"### {fleet.title()} findings", '',
                         f"- The busiest observed pickup hour was {int(busiest['pickup_hour']):02d}:00, with {int(busiest['count']):,} accepted trips across the period.",
                         f"- Accepted trips totaled {total_2020:,} in 2020 and {total_2024:,} in 2024. These describe this cleaned TLC population, not all NYC transport demand.",
                         f"- Median metered fare was ${float(fare_summary['50%']):.2f}; the 75th percentile was ${float(fare_summary['75%']):.2f}, while the maximum was ${float(fare_summary['max']):,.2f}. The tail warrants separate evaluation.",
                         f"- Among records explicitly coded as card or cash, the most frequent payment method was {payment_label}. Other entries describe fare regime, payment status or missing/unknown information, and remain in the analysis.", ''])
    summary += ['', 'The population is trips with positive **recorded fares**, not verified settled payments. Codes 0 and 3–6 and missing payment information remain when the other cleaning rules pass; payment type is not a model input. Zero/negative recorded fares are excluded regardless of payment code. See [payment definitions](../data.md) for the versioned meaning of code zero.', '', 'Training: 2020–2022. Validation: 2023. Test: 2024. Starting fractions: yellow 0.002, green 0.05; seed 42. '
                'The training-size comparison uses fixed validation/test samples. The selected training count can be larger than the starting count.', '',
                '## Held-out performance', ''] + metrics_table
    summary += ['', 'Only the final selected candidate is scored on the held-out test set. '
                'The target is the completed-trip metered fare, not total passenger spending or a pre-trip quote.', '',
                '## Sample-size sensitivity', '',
                '| Fleet | Training rows | Training fraction | Validation RMSE ($) | Validation MAE ($) |',
                '|---|---:|---:|---:|---:|']
    for fleet, run in experiment['fleets'].items():
        for candidate in run['modeling']['sample_size_comparison']:
            summary.append(f"| {fleet.title()} | {candidate['training_rows']:,} | {candidate['training_fraction']:.4f} | "
                           f"{candidate['validation']['rmse']:.2f} | {candidate['validation']['mae']:.2f} |")
    summary += ['', 'The larger candidate uses the same selected regression family, with preprocessing fitted again on the larger training set. '
                'This assesses the complete training procedure at two sizes; it does not isolate sample count from changes in learned vocabularies and caps.', '',
                '## Extreme-error investigation', '',
                '| Fleet | Uncapped linear validation RMSE ($) | Capped linear validation RMSE ($) | Worst 20 share of uncapped squared error |',
                '|---|---:|---:|---:|']
    for fleet, run in experiment['fleets'].items():
        m = run['modeling']
        capped = next(row for row in m['validation'] if row['model']=='LinearRegression')
        share = m['uncapped_linear_diagnostics']['worst_20_share_of_squared_error']
        summary.append(f"| {fleet.title()} | {m['uncapped_linear_validation']['rmse']:.2f} | {capped['rmse']:.2f} | {share:.1%} |")
    summary += ['', 'Distance/duration feature inputs are capped at training-set 99.9th percentiles. Raw records and fare labels are retained. '
                'The uncapped comparison uses identical base train/validation records. The table measures sensitivity to the caps; '
                'it does not prove that every extreme record is erroneous. Worst-residual CSVs expose the measurements for inspection.', '',
                '## Analytical findings', ''] + findings
    investigation_source = run_root / 'verification-error-investigation'
    if (investigation_source / 'investigation.json').exists():
        investigation_target = destination / 'verification-error-investigation'
        investigation_target.mkdir(exist_ok=True)
        for artifact in investigation_source.iterdir():
            if artifact.suffix in ('.json', '.csv'):
                shutil.copy2(artifact, investigation_target / artifact.name)
        investigation = json.loads((investigation_target / 'investigation.json').read_text())
        uncapped = investigation['variants']['uncapped']
        capped = investigation['variants']['training_only_caps']
        worst = rows(investigation_target / 'uncapped_worst_errors.csv')[0]
        summary += ['## Reproducing the earlier verification failure', '',
                    'A separate January–March 2024 diagnostic rerun reproduces the earlier yellow linear-regression failure on exactly the same split counts: '
                    f"{investigation['split_counts']['train']:,} training and {investigation['split_counts']['validation']:,} validation records.", '',
                    f"The largest residual came from a record with **{float(worst['trip_distance']):,.1f} miles in {float(worst['duration_minutes']):.1f} minutes**, "
                    f"an actual fare of **${float(worst['fare_amount']):.2f}**, and an uncapped prediction of **${float(worst['prediction']):,.2f}**. "
                    'Those measurements are highly inconsistent with an ordinary taxi trip and caused extreme linear extrapolation.', '',
                    f"On identical train/validation records, training-only input caps changed linear-regression validation RMSE from "
                    f"**${uncapped['metrics']['rmse']:.2f} to ${capped['metrics']['rmse']:.2f}**. "
                    f"The worst 20 records contributed {uncapped['diagnostics']['worst_20_share_of_squared_error']:.1%} of uncapped squared error. "
                    'This is a diagnostic comparison, not the main five-year performance result and not a test-score claim.', '',
                    'Reproduce with `python -m nyc_taxi.investigate`. Compact evidence is in `verification-error-investigation/`. '
                    'Its additional runtime is recorded separately in `investigation.json`.', '']
    summary += ['## Temporal pricing context', '',
                'TLC increased the taxi/SHL meter fare structure effective 19 December 2022, near the end of the training period. '
                '[Official TLC notice](https://www.nyc.gov/assets/tlc/downloads/pdf/industry-notices/industry_notice_22_02_english.pdf). '
                'Most training trips therefore precede the new price structure, whereas validation and test trips follow it. '
                'This is a plausible contributor to underestimation visible in the plots, not a measured causal attribution. '
                'An explicit pricing-regime feature or a rolling retraining experiment is a future extension; neither was included in these scores.', '']
    summary += ['## Measured local runtime', '',
                f"Processing wall time: **{experiment['elapsed_seconds']/60:.1f} minutes**, excluding download and notebook rendering.", '',
                '| Fleet | Load/clean/Parquet (min) | SQLite snapshot (min) | EDA (min) | Modeling (min) |',
                '|---|---:|---:|---:|---:|']
    for fleet, run in experiment['fleets'].items():
        t=run['stage_seconds']
        summary.append(f"| {fleet.title()} | {t['load_clean_and_parquet_seconds']/60:.1f} | {t['sqlite_seconds']/60:.1f} | "
                       f"{t['eda_seconds']/60:.1f} | {t['modeling_seconds']/60:.1f} |")
    env=experiment['environment']
    summary += ['', f"Environment: Python {env['python'].split()[0]}, Spark {env['spark']}, {env['master']}, 4 GB driver setting, New York timezone. "
                'Complete Java and operating-system details are in `experiment.json`. No clustered benchmark is claimed.', '',
                '## Storage and limitations', '',
                'All accepted trips are persisted in cleaned Parquet. SQLite is a bounded snapshot of at most 100,000 rows per fleet; '
                'it is not the five-year data warehouse and is not used for modeling or EDA. Full export is configurable when disk space permits.', '',
                'These are fixed-seed local results, with modest fixed model settings and one temporal holdout. '
                'They do not quantify uncertainty across repeated samples. Filtering nonpositive recorded fares excludes zero/negative fare values but does not necessarily exclude trips coded no charge or dispute, '
                'and source data accuracy is not guaranteed. Feature caps can suppress real long-distance behavior; fare-band diagnostics '
                'and full-range plots make remaining tail errors visible.', '',
                'Prediction plots use up to 2,000 seeded hash-selected test records. The zoomed view excludes fares beyond $100 visually, '
                'but the full view and metrics retain them. Drop-off rankings require at least 500 trips.', '',
                '## Charts', '',
                '![Yellow demand](yellow/demand.png)', '![Green demand](green/demand.png)',
                '![Yellow held-out predictions](yellow/actual_vs_predicted.png)',
                '![Green held-out predictions](green/actual_vs_predicted.png)', '']
    (destination / 'README.md').write_text('\n'.join(summary))
    return experiment, '\n'.join(metrics_table)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('outputs/full-five-year/real'))
    parser.add_argument('--destination', type=Path, default=Path('docs/results'))
    args=parser.parse_args()
    publish(args.run,args.destination)


if __name__ == '__main__':
    main()
