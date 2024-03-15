
def unnamed(inference_lgbm_hyperopt_featImp_quantilePerf):

    n_col = "n"
    ref_dt = 'reference_date_m'
    target_col = 'target'
    true_positives_col = "true_positives"
    n_cut_col = "n_cut"
    cum_true_positives_col = "cum_true_positives"
    percent_rank_col = "percent_rank"
    temp_percentile_col = "percentile_"
    percentile_col = 'percentile'
    quantile_granularity = 100
    recall_col = 'recall'
    precision_col = 'precision'
    f1_col = 'f1_score'
    uplift_col = 'uplift'
    baseline_incidence = 'baseline_incidence'
    overall_samples = 'overall_ndgs'
    overall_pos = 'overall_pos'

    df = (
        inference_df
        .select(
            'ndg',
            ref_dt, 
            target_col,
            prediction_col,
            F.percent_rank().over(Window.partitionBy('reference_date_m').orderBy(prediction_col)).alias(percent_rank_col),
        )
        .withColumn(temp_percentile_col,(F.col(percent_rank_col) * quantile_granularity).cast(IntegerType()))
        .withColumn(percentile_col,
            F.when(
                F.col(temp_percentile_col) == quantile_granularity,
                F.lit(quantile_granularity - 1),
            ).otherwise(F.col(temp_percentile_col)),
        )
        .drop(temp_percentile_col)
    )

    percentile_counts = (
        df
        .groupby(ref_dt, percentile_col)
        .agg(
            F.count("*").alias(n_col), 
            F.sum(F.col(target_col)).alias(true_positives_col)
        )
    )

    w = Window.partitionBy(ref_dt).orderBy(F.desc(percentile_col))
    w_date = Window.partitionBy(ref_dt)
    metrics_df = (
        percentile_counts
        .withColumn(n_cut_col, F.sum(F.col(n_col)).over(w))
        .withColumn(
            cum_true_positives_col, F.sum(F.col(true_positives_col)).over(w)
        )
        .withColumn(
            recall_col, F.col(cum_true_positives_col) / F.sum(F.col(true_positives_col)).over(Window.partitionBy(ref_dt))
        )
        .withColumn(
            precision_col, F.col(cum_true_positives_col) / F.col(n_cut_col)
        )
        .withColumn(
            f1_col,
            2.0
            * (F.col(precision_col) * F.col(recall_col))
            / (F.col(precision_col) + F.col(recall_col)),
        )
        .withColumn(overall_samples, F.sum(F.col(n_col)).over(w_date))
        .withColumn(overall_pos, F.sum(F.col(true_positives_col)).over(w_date))
        .withColumn(baseline_incidence, F.col(overall_pos)/F.col(overall_samples))
        .withColumn(uplift_col, F.col(precision_col)/F.col(baseline_incidence))
        .orderBy(F.col(percentile_col).desc())
    )



# Functions defined here will be available to call in
# the code for any table.
from pyspark.sql import Window, functions as F
from pyspark.sql import types as T
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import QuantileDiscretizer
import datetime
import itertools
from sklearn.metrics import auc, precision_recall_curve, recall_score, precision_score, roc_auc_score
import pickle
import pandas as pd

Q = 90
METRICS = [
    'recall',
    'precision',
    'baseline_uplift',
    'target_uplift'
    ]

schema = StructType(
        [
            StructField('reference_date_m', DateType(), False),
            StructField('partition', StringType(), False),
            StructField(f'{Q}_quantile_recall', FloatType(), True),
            StructField(f'{Q}_quantile_precision', FloatType(), True),
            StructField('auc', FloatType(), True),
            StructField('detected_positives', IntegerType(), False),
            StructField('tot_positives', IntegerType(), False),
            StructField('mean_target', FloatType(), False),
            StructField('mean_pred', FloatType(), False),
            StructField('count', IntegerType(), False),
            StructField(f'{Q}_count', IntegerType(), True)
        ]
    )

# PandasUDF that trains the lgb Classifier on a specific fold, on the train partition and the applies on the val set
def evaluate_fold(group):
    from sklearn.metrics import roc_auc_score
    import pandas as pd  # noqa

    group = group.sort_values('pred')
    # create a (unique) ranking columns that follows the sorting computed above
    group['rank'] = range(group.shape[0])
    # Compute quantiles over the ranking column, in order to group the predictions
    group['quantile'] = pd.qcut(group['rank'], 1000, labels=False)

    if group['target'].sum() > 1:
        auc = roc_auc_score(group['target'], group['pred'])
    else:
        auc = None
    mean_target = group['target'].mean()
    mean_pred = group['pred'].mean()

    # Compute metrics with such support:
    # Detected Positive: all the samples included in the top quantile(s)
    # True Positive: all the samples with target = 1 in the top quantile(s)
    # Tot Positive: all the samples with target = 1 in the validation set
    tot_pos = group['target'].sum()
    best_quantile = group[group['quantile'] >= Q]['target']
    detected_pos = best_quantile.sum()
    # Recall = Detected Positive / Tot Positive
    recall = detected_pos / tot_pos
    # Precision = True Positive / Detected Positive
    precision = best_quantile.mean()

    results_df = pd.DataFrame(
        [
            {
                'reference_date_m': group['reference_date_m'].values[0],
                'partition': group['partition'].values[0],
                f'{Q}_quantile_recall': recall,
                f'{Q}_quantile_precision': precision,
                'auc': auc,
                'detected_positives': detected_pos,
                'tot_positives': tot_pos,
                'mean_target': mean_target,
                'mean_pred': mean_pred,
                'count': group.shape[0],
                f'{Q}_count': len(best_quantile)
            }
        ]
    )
    return results_df

def AUPRC(y_true, y_pred, pos_label=1):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label)
    return auc(recall, precision)

def AUPRC_lgbm(y_true, y_pred):
    '''
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    Custom eval function expects a callable with following signatures:
        - func(y_true, y_pred)
        - func(y_true, y_pred, weight)
        - func(y_true, y_pred, weight, group)
    returns:
        - (eval_name: str, eval_result: float, is_higher_better: bool)
        - list of (eval_name: str, eval_result: float, is_higher_better: bool)
    '''
    return 'AUPRC', AUPRC(y_true, y_pred), True

def metrics_backbone_calculation(inference_df, prediction_col):
        n_col = "n"
        ref_dt = 'reference_date_m'
        target_col = 'target'
        true_positives_col = "true_positives"
        n_cut_col = "n_cut"
        cum_true_positives_col = "cum_true_positives"
        percent_rank_col = "percent_rank"
        temp_percentile_col = "percentile_"
        percentile_col = 'percentile'
        quantile_granularity = 100
        recall_col = 'recall'
        precision_col = 'precision'
        f1_col = 'f1_score'
        uplift_col = 'uplift'
        baseline_incidence = 'baseline_incidence'
        overall_samples = 'overall_ndgs'
        overall_pos = 'overall_pos'

        df = (
            inference_df
            .select(
                'ndg',
                ref_dt, 
                target_col,
                prediction_col,
                F.percent_rank().over(Window.partitionBy('reference_date_m').orderBy(prediction_col)).alias(percent_rank_col),
            )
            .withColumn(temp_percentile_col,(F.col(percent_rank_col) * quantile_granularity).cast(IntegerType()))
            .withColumn(percentile_col,
                F.when(
                    F.col(temp_percentile_col) == quantile_granularity,
                    F.lit(quantile_granularity - 1),
                ).otherwise(F.col(temp_percentile_col)),
            )
            .drop(temp_percentile_col)
        )

        percentile_counts = (
            df
            .groupby(ref_dt, percentile_col)
            .agg(
                F.count("*").alias(n_col), 
                F.sum(F.col(target_col)).alias(true_positives_col)
            )
        )

        w = Window.partitionBy(ref_dt).orderBy(F.desc(percentile_col))
        w_date = Window.partitionBy(ref_dt)
        metrics_df = (
            percentile_counts
            .withColumn(n_cut_col, F.sum(F.col(n_col)).over(w))
            .withColumn(
                cum_true_positives_col, F.sum(F.col(true_positives_col)).over(w)
            )
            .withColumn(
                recall_col, F.col(cum_true_positives_col) / F.sum(F.col(true_positives_col)).over(Window.partitionBy(ref_dt))
            )
            .withColumn(
                precision_col, F.col(cum_true_positives_col) / F.col(n_cut_col)
            )
            .withColumn(
                f1_col,
                2.0
                * (F.col(precision_col) * F.col(recall_col))
                / (F.col(precision_col) + F.col(recall_col)),
            )
            .withColumn(overall_samples, F.sum(F.col(n_col)).over(w_date))
            .withColumn(overall_pos, F.sum(F.col(true_positives_col)).over(w_date))
            .withColumn(baseline_incidence, F.col(overall_pos)/F.col(overall_samples))
            .withColumn(uplift_col, F.col(precision_col)/F.col(baseline_incidence))
            .orderBy(F.col(percentile_col).desc())
        )

        return metrics_df

