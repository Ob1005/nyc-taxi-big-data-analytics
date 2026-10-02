"""Publish findings directly from the latest real run; never type result values."""
import json
import shutil
from pathlib import Path


def read(path): return json.loads(Path(path).read_text())


def model_table(experiment):
    lines = ['| Fleet / setting | Selected model | MAE ($) | RMSE ($) | R² | Within $2 | Within $5 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for fleet, run in experiment['fleets'].items():
        model = run['modeling']
        cases = [('Mean baseline', 'Training mean', model['mean_baseline_test'])]
        cases += [(name.replace('_','-'), r['selected_model'].replace('Regressor',''), r['test']) for name,r in model['feature_sets'].items()]
        for name, estimator, m in cases:
            r2 = f"{m['r2']:.3f}" if m['r2'] is not None else 'undefined'
            lines.append(f"| {fleet.title()} / {name} | {estimator} | {m['mae']:.2f} | {m['rmse']:.2f} | {r2} | {m['within_2_pct']:.1f}% | {m['within_5_pct']:.1f}% |")
    return '\n'.join(lines)


def findings(root):
    monthly = read(root/'yellow/eda/monthly_trips.json')
    yearly = {}
    for r in monthly: yearly[r['month'][:4]] = yearly.get(r['month'][:4],0)+r['trip_count']
    low = min(monthly,key=lambda r:r['trip_count'])
    january = next(r for r in monthly if r['month']=='2020-01')
    hour = max(read(root/'yellow/eda/hourly_trips.json'),key=lambda r:r['trip_count'])
    bands = read(root/'yellow/eda/fare_by_distance.json')
    zones = read(root/'yellow/eda/highest_fare_dropoffs.json'); zone=zones[0]
    return [
        (f"Yellow accepted pickups fell from {january['trip_count']:,} in January 2020 to {low['trip_count']:,} in {low['month']}, the lowest observed month ({(1-low['trip_count']/january['trip_count'])*100:.1f}% lower). Annual volume rose {(yearly['2024']/yearly['2020']-1)*100:.1f}% between 2020 and 2024.", 'monthly_trips'),
        (f"The busiest yellow pickup hour was {hour['pickup_hour']:02d}:00–{hour['pickup_hour']:02d}:59 New York time, with {hour['trip_count']:,} accepted trips across the study.", 'hourly_trips'),
        (f"Yellow mean recorded fare was ${bands[0]['mean_fare']:.2f} in the {bands[0]['distance_band']}-mile band and ${bands[-1]['mean_fare']:.2f} in the {bands[-1]['distance_band']}-mile band. These are descriptive averages, mixing routes and pricing periods.", 'fare_by_distance'),
        (f"Destination zone {zone['DOLocationID']} had the highest yellow mean fare (${zone['mean_fare']:.2f}, {zone['trip_count']:,} trips) among mapped zones with at least 500 accepted trips. This does not adjust for trip length.", 'highest_fare_dropoffs'),
    ]


def publish(root):
    root = Path(root); experiment = read(root/'experiment.json')
    destination = Path('docs/results')
    # Replace the published run as a unit; old result files must not survive a new study.
    if destination.exists(): shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(root/'experiment.json', destination/'experiment.json')
    for fleet in experiment['fleets']:
        target=destination/fleet; target.mkdir()
        for name in ('quality.json','data_quality_examples.json','run.json'):
            shutil.copy2(root/fleet/name,target/name)
        for subdir in ('eda','modeling'):
            for p in (root/fleet/subdir).glob('*'):
                if p.suffix in ('.json','.png'):
                    shutil.copy2(p,target/p.name)
    insights = findings(root)
    table = model_table(experiment)
    minutes=experiment['elapsed_seconds']/60
    yellow=experiment['fleets']['yellow']['modeling']
    pre=yellow['feature_sets']['pre_trip']['test']['rmse']
    post=yellow['feature_sets']['post_trip']['test']['rmse']
    baseline=yellow['mean_baseline_test']['rmse']
    baseline_change=(1-pre/baseline)*100
    post_change=(1-post/pre)*100
    interpretation=f"Yellow pre-trip RMSE was {abs(baseline_change):.1f}% {'below' if baseline_change>=0 else 'above'} its mean baseline; post-trip RMSE was {abs(post_change):.1f}% {'lower' if post_change>=0 else 'higher'} than pre-trip RMSE."
    counts='; '.join(f"{fleet}: {r['quality']['accepted_rows']:,} accepted trips" for fleet,r in experiment['fleets'].items())
    fractions=' and '.join(f"{r['modeling']['sample_fraction']*100:g}% {fleet}" for fleet,r in experiment['fleets'].items())
    samples='; '.join(f"{fleet}: {m['modeling']['split_counts']['train']:,} train / {m['modeling']['split_counts']['validation']:,} validation / {m['modeling']['split_counts']['test']:,} test" for fleet,m in experiment['fleets'].items())
    readme=f'''# NYC taxi fare analytics

Estimate NYC taxi fares before travel and compare those estimates with reconstruction from completed trips.

Started as coursework in my MSc big data course; extended into a full five-year pipeline.

```mermaid
flowchart LR
    A[TLC monthly Parquet] --> B[Normalize and plausibility filters]
    B --> C[Cleaned Parquet]
    C --> D[Spark SQL demand and fare analysis]
    C --> E[Seeded samples and temporal split]
    E --> F[Training-only route medians and encoding]
    F --> G[Pre-trip and post-trip model comparison]
    G --> H[Validation selection then test evaluation]
    D --> I[Saved results and notebook]
    H --> I
```

## Key findings: yellow taxis

'''
    for sentence, chart in insights:
        readme+=f'{sentence}\n\n![{chart.replace("_"," ")}](docs/results/yellow/{chart}.png)\n\n'
    readme+=f'''## Model results

{table}

{interpretation}
Both settings assume a known destination; post-trip also uses realized distance and duration, making it a reconstruction upper bound.
The known December 2022 fare-change flag is included in both settings; this comparison does not isolate its contribution or establish a causal effect.

The full local run took **{minutes:.2f} minutes** with Spark {experiment['environment']['spark']}, {experiment['environment']['master']} and a {experiment['environment']['driver_memory']} driver: {counts}. Modelling uses seeded samples for local-machine compute: {fractions}; {samples}.

## How to run

Python 3.9–3.12 and Java 17 are required. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
nyc-taxi demo
pytest -q
nyc-taxi download
nyc-taxi run
```

Open [the notebook](notebooks/nyc_taxi_portfolio.ipynb) in Jupyter or VS Code and run its cells in order. It reads saved results by default; set `RECOMPUTE = True` to run the full pipeline. [Methodology](docs/methodology.md) explains the assumptions; [results](docs/results/README.md) contains the counts, comparisons and diagnostics.

## Limitations

- Local Spark demonstrates distributed APIs on one machine.
- One temporal holdout; no rolling evaluation.
- Model training and evaluation use samples, not every accepted trip.
- Positive recorded fares do not establish that payment occurred; filters exclude some legitimate unusual trips.
- Pre-trip route medians omit traffic conditions; post-trip inputs require completed trips.
- No deployed service or production monitoring.
'''
    Path('README.md').write_text(readme)
    report=f'''# Results from the current real run

These files were generated by `nyc-taxi run`. Tables use JSON; cleaned records and fitted route summaries use Parquet in the locally ignored data/output folders. Model artifacts remain local.

Runtime: {minutes:.2f} minutes. Environment: Spark {experiment['environment']['spark']}; {experiment['environment']['master']}; driver {experiment['environment']['driver_memory']}; New York session timezone.

{counts}.

{table}

## Cleaning accounting

Each row receives its first failing reason. Counts below are mutually exclusive, so source = accepted + rejected. Rule order and rationale are in [methodology](../methodology.md).

| Fleet | Reason | Rows |
|---|---|---:|
'''
    order=['missing_timestamp','outside_period','distance_missing_or_nonpositive','distance_over_limit','duration_too_short','duration_too_long','speed_over_limit','fare_missing_or_nonpositive','fare_over_limit','accepted']
    for fleet,r in experiment['fleets'].items():
        q=r['quality']
        report+=f"| {fleet} | source total | {q['source_rows']:,} |\n"
        for reason in order: report+=f"| {fleet} | {reason} | {q['first_failure_counts'].get(reason,0):,} |\n"
    report+='\n## Validation comparisons\n\nWinners were selected separately for each feature setting by validation RMSE. The test set was not used for selection.\n\n| Fleet | Setting | Model | Validation RMSE ($) |\n|---|---|---|---:|\n'
    for fleet,r in experiment['fleets'].items():
        m=r['modeling']
        for setting,s in m['feature_sets'].items():
            for v in s['validation']: report+=f"| {fleet} | {setting} | {v['model']} | {v['rmse']:.3f} |\n"
    report+='\n## Sample and route coverage\n\n'+samples+'.\n\n'
    for fleet,r in experiment['fleets'].items():
        m=r['modeling']; flag={int(v['new_fare_regime']):v['count'] for v in m['training_pricing_regimes']}
        report+=f"{fleet.title()}: {m['test_unseen_pair_rows']:,} test rows used the unseen-route fallback. Training contained {flag.get(0,0):,} pre-change and {flag.get(1,0):,} post-change rows.\n\n"
    report+='## Data quality findings\n\n'
    for fleet,r in experiment['fleets'].items():
        for e in r['quality']['examples']:
            report+=f"The source {fleet} record at {e['pickup_datetime']} reports {e['trip_distance']:,.1f} miles in {e['duration_minutes']:.2f} minutes with a ${e['fare_amount']:.2f} fare. It is rejected as `{e['quality_reason']}`. This example illustrates why physically implausible inputs are removed before modelling.\n\n"
    report+='## Error diagnostics\n\nEach fleet has `pre_trip_error_by_fare.json` and `post_trip_error_by_fare.json`, plus the 20 largest absolute test residuals for each setting. Inspect these alongside the averages; the test population has already passed the cleaning rules. See [methodology](../methodology.md) for interpretation.\n'
    (destination/'README.md').write_text(report)
