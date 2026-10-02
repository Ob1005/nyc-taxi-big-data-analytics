"""Reproduce the three-month verification failure and assess input-cap sensitivity."""
import argparse
import json,time
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.regression import LinearRegression
from nyc_taxi.pipeline import load,clean,features,fit_preprocessing,regression_metrics,residual_diagnostics
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('outputs/full-five-year/real/verification-error-investigation'))
    args = parser.parse_args()
    started=time.perf_counter()
    root=args.output; root.mkdir(parents=True, exist_ok=True)
    spark=(SparkSession.builder.master('local[2]').config('spark.driver.memory','4g').config('spark.sql.shuffle.partitions','8').config('spark.sql.session.timeZone','America/New_York').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        raw=load(spark,args.data,'yellow',years=[2024],months=[1,2,3])
        accepted,quality,tagged=clean(raw)
        enriched=features(accepted).cache()
        sample=enriched.sample(False,0.01,42).cache()
        training=sample.filter(F.col('pickup_datetime')<'2024-02-01').cache()
        validation=sample.filter((F.col('pickup_datetime')>='2024-02-01') & (F.col('pickup_datetime')<'2024-03-01')).cache()
        counts={'train':training.count(),'validation':validation.count()}
        scores={}
        for name,capped in [('uncapped',False),('training_only_caps',True)]:
            preprocessor,caps=fit_preprocessing(training,cap_inputs=capped)
            model=LinearRegression(labelCol='fare_amount',regParam=0.1).fit(preprocessor.transform(training))
            predictions=model.transform(preprocessor.transform(validation)).cache()
            scores[name]={'metrics':regression_metrics(predictions),'caps':caps,'diagnostics':residual_diagnostics(predictions,root,name)}
            predictions.unpersist()
        report={'source_months':['2024-01','2024-02','2024-03'],'quality':quality,'split_counts':counts,'sample_fraction':0.01,'seed':42,'variants':scores,'elapsed_seconds':time.perf_counter()-started}
        (root/'investigation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
