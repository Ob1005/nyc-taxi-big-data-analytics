"""Build the single walkthrough from the current pipeline source and saved results."""
from pathlib import Path
import inspect
import nbformat as nbf
from nyc_taxi import pipeline

n = nbf.v4.new_notebook()
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
def show(function): code(inspect.getsource(function))

md('# NYC taxi fare analytics\n\nCan we estimate a taxi fare before travel, and how does that compare with reconstructing a completed trip? Yellow is the main fleet; green is a shorter companion comparison. This walkthrough shows the implementation and reads results from the actual full run. See [methodology](../docs/methodology.md) for assumptions.')
code('''from pathlib import Path
import json, subprocess, sys
from IPython.display import display, Markdown, Image
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from nyc_taxi.config import RULES, FARE_CHANGE_DATE, SEED
from nyc_taxi.pipeline import CATEGORICAL, FEATURE_SETS, EDA_QUERIES
from nyc_taxi.reporting import model_table
ROOT = Path.cwd()
if not (ROOT / 'pyproject.toml').exists(): ROOT = ROOT.parent
RECOMPUTE = False
if RECOMPUTE:
    subprocess.run([sys.executable, '-m', 'nyc_taxi.cli', 'run'], cwd=ROOT, check=True)
RESULTS = ROOT / 'docs/results'
def read(path): return json.loads((RESULTS / path).read_text())
def table(rows):
    if not rows: return
    keys = list(rows[0])
    display(Markdown('| ' + ' | '.join(keys) + ' |\\n| ' + ' | '.join(['---']*len(keys)) + ' |\\n' +
                     '\\n'.join('| ' + ' | '.join(str(r.get(k,'')) for k in keys) + ' |' for r in rows)))
experiment = read('experiment.json')
''')
md('## 1. Source files\n\nEvery monthly yellow and green TLC Parquet file in the study period is required. Each file is projected and cast separately before union, allowing monthly numeric type differences without guessing a combined schema.')
show(pipeline.normalize)
code("table([{'fleet': f, 'source_files': len(r['source_files']), 'source_rows': r['quality']['source_rows']} for f,r in experiment['fleets'].items()])")
md('The source totals describe loaded records, before any population restrictions. Full-data analysis and sampled modelling therefore have different row counts.')
md('## 2. Cleaning and data quality\n\nThe following is the actual first-failure filter used by the run. Thresholds are defined once in the configuration file; they define ordinary plausible NYC taxi journeys.')
show(pipeline.cleaning_reason)
show(pipeline.clean)
code("from dataclasses import asdict\ntable([asdict(RULES)])\nfor fleet, r in experiment['fleets'].items():\n    display(Markdown('### '+fleet.title()))\n    q=r['quality']\n    table([{'reason': k, 'rows': v} for k,v in q['first_failure_counts'].items()])\n    assert sum(q['first_failure_counts'].values()) == q['source_rows']\n    table(q['examples'])")
md('### Data quality findings\n\nThe source example above is rejected by the distance limit before modelling. Rejection counts are mutually exclusive: a row failing several rules contributes only to its first failure. These assumptions can exclude legitimate unusual journeys; accepted records are positive-recorded-fare trips, not proof of payment.')
md('## 3. Parquet storage and Spark SQL\n\nThe full run writes accepted records to Parquet, then reads them back. This breaks the long monthly input plan and provides a reusable cleaned layer.')
md('```python\naccepted.write.mode("overwrite").parquet(str(cleaned_path))\naccepted = spark.read.parquet(str(cleaned_path))\nenriched = features(accepted)\nenriched.createOrReplaceTempView("trips")\n```')
code("table([{'fleet': f, 'accepted_rows': r['quality']['accepted_rows'], 'cleaning_seconds': round(r['stage_seconds']['cleaning'],2)} for f,r in experiment['fleets'].items()])")
md('Spark SQL operates on every accepted trip for the following descriptive analysis. Only the smaller modelling sample is cached in memory.')
md('## 4. Exploratory analysis\n\nThese are three of the SQL queries executed on the cleaned trips. The distance-band column is derived only for analysis; it is not an extra model input.')
for name in ('monthly_trips','fare_by_distance','highest_fare_dropoffs'):
    md('```sql\n'+pipeline.EDA_QUERIES[name].format(minimum=500)+'\n```')
code("""monthly=read('yellow/monthly_trips.json')
years={}
for r in monthly: years[r['month'][:4]]=years.get(r['month'][:4],0)+r['trip_count']
low=min(monthly,key=lambda r:r['trip_count'])
hour=max(read('yellow/hourly_trips.json'),key=lambda r:r['trip_count'])
bands=read('yellow/fare_by_distance.json')
zone=read('yellow/highest_fare_dropoffs.json')[0]
interpretations=[
    f"Accepted yellow pickups: {years['2020']:,} in 2020 and {years['2024']:,} in 2024. Lowest observed month: {low['month']} ({low['trip_count']:,}). No pre-pandemic year is available.",
    f"Busiest yellow pickup hour: {hour['pickup_hour']}:00, with {hour['trip_count']:,} trips over the study.",
    f"Mean recorded fare rises from ${bands[0]['mean_fare']:.2f} in the {bands[0]['distance_band']}-mile band to ${bands[-1]['mean_fare']:.2f} in the {bands[-1]['distance_band']}-mile band. This mixes routes and pricing periods.",
    f"Zone {zone['DOLocationID']} leads mean fare at ${zone['mean_fare']:.2f} over {zone['trip_count']:,} trips. The ranking does not control for distance."
]
for name, text in zip(EDA_QUERIES,interpretations):
    display(Image(filename=str(RESULTS/'yellow'/f'{name}.png')))
    display(Markdown(text))
display(Markdown('### Green companion'))
display(Image(filename=str(RESULTS/'green/monthly_trips.png')))
""")
md('The charts describe the filtered source population. Demand begins in the first year of the pandemic, and route length can explain much of the destination-fare ranking; these are descriptive findings.')
md('## 5. Features and temporal split\n\nTraining uses the earlier period, validation chooses a model, and the final year is held out. Both settings know the destination. Route summaries and category vocabularies learn from the training sample only; unseen pairs fall back to overall training medians. Pre-trip features exclude that trip’s realized distance and duration.')
show(pipeline.features)
show(pipeline.fit_history)
show(pipeline.add_history)
show(pipeline.fit_preprocessing)
code("for fleet,r in experiment['fleets'].items():\n    m=r['modeling']\n    table([{'fleet':fleet,'fraction':m['sample_fraction'],**m['split_counts'],'train_before':m['train_before'],'test_from':m['test_from'],'test_unseen_pairs':m['test_unseen_pair_rows']}])\n    table(m['training_pricing_regimes'])\n    table([m['fallback']])\ntable([{'setting': k, 'numeric_features': ', '.join(v)} for k,v in FEATURE_SETS.items()])")
md('The external fare-change flag becomes true on December 19, 2022. It supplies known pricing information but only a short new-regime window is present in training. This run does not isolate the flag’s accuracy contribution. Post-trip results use additional information unavailable before travel.')
md('## 6. Model comparison and final evaluation\n\nThe following excerpt is the actual candidate selection loop. Each feature setting compares linear regression, random forest and GBT by validation RMSE. The training-mean baseline and the two winners are evaluated on the same test rows.')
source=inspect.getsource(pipeline.train)
start=source.index('            estimators =')
end=source.index('            residual_diagnostics')
md('```python\n'+inspect.cleandoc(source[start:end])+'\n```')
show(pipeline.regression_metrics)
code("for fleet,r in experiment['fleets'].items():\n    display(Markdown('### '+fleet.title()))\n    table([{'setting':s,**v} for s,m in r['modeling']['feature_sets'].items() for v in m['validation']])\ndisplay(Markdown(model_table(experiment)))\nfor fleet in experiment['fleets']:\n    for setting in FEATURE_SETS:\n        display(Markdown(f'### {fleet.title()}: {setting} errors by fare band'))\n        table(read(f'{fleet}/{setting}_error_by_fare.json'))\n        if fleet=='yellow':\n            display(Markdown('Largest absolute test residuals after filtering'))\n            table(read(f'{fleet}/{setting}_worst_errors.json'))\ndisplay(Markdown(f\"Full run: {experiment['elapsed_seconds']/60:.2f} minutes. Environment: Spark {experiment['environment']['spark']}, {experiment['environment']['master']}, {experiment['environment']['driver_memory']} driver.\"))")
code("y=experiment['fleets']['yellow']['modeling']\npre=y['feature_sets']['pre_trip']; post=y['feature_sets']['post_trip']\ndisplay(Markdown(f\"Yellow selected {pre['selected_model']} for pre-trip and {post['selected_model']} for post-trip. Test RMSE: ${pre['test']['rmse']:.2f} pre-trip versus ${post['test']['rmse']:.2f} post-trip; mean baseline ${y['mean_baseline_test']['rmse']:.2f}. The same held-out trips support this comparison.\"))")
md('Compare pre-trip estimates with the mean baseline first; post-trip reconstruction answers an easier question. Fare-band errors and the largest residuals expose remaining failures after cleaning. Selection used validation, not these test diagnostics. See [results](../docs/results/README.md) for all published tables and [methodology](../docs/methodology.md) for limitations.')
n['cells']=cells
n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
nbf.validate(n)
nbf.write(n,Path('notebooks/nyc_taxi_portfolio.ipynb'))
