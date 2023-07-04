from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F, Window
import cards_propensity_experiments.utils.experiment_manager as exp_manager
from cards_propensity_experiments.utils.experiment_manager import N_Q
from pyspark.sql.types import IntegerType


def prediction_generator(config_dict, baselines):

    transforms = []

    for _, experiment in config_dict.items():

        for model_name in baselines:

            model_exp = baselines[model_name]

            @configure(profile=[
                'AUTO_BROADCAST_JOIN_DISABLED',
                'EXECUTOR_MEMORY_LARGE',
                'EXECUTOR_MEMORY_OVERHEAD_EXTRA_LARGE',
                'NUM_EXECUTORS_32',
                'DRIVER_MEMORY_EXTRA_LARGE',
                'DRIVER_MEMORY_OVERHEAD_EXTRA_EXTRA_LARGE',
                'SHUFFLE_PARTITIONS_LARGE',
                'ARROW_ENABLED'])
            @transform(
                metrics=Output(f"{experiment['output_location']}/inference/{exp_manager.EXPERIMENT_NAME}_{model_name}_metrics"),
                predictions=Input(f"{experiment['output_location']}/inference/{exp_manager.EXPERIMENT_NAME}_{model_name}_predictions", branch="baseline_models")
            )
            def compute_model_metrics(
                ctx,
                metrics,
                predictions,
                model_dict=model_exp
            ):

                n_col = "n"
                ref_dt = 'reference_date_m'
                target_col = 'target'
                true_positives_col = "true_positives"
                n_cut_col = "n_cut"
                cum_true_positives_col = "cum_true_positives"
                percent_rank_col = "percent_rank"
                temp_percentile_col = "percentile_"
                percentile_col = 'percentile'
                prediction_col = 'pred'
                quantile_granularity = N_Q
                recall_col = 'recall'
                precision_col = 'precision'
                f1_col = 'f1_score'
                uplift_col = 'uplift'
                baseline_incidence = 'baseline_incidence'
                overall_samples = 'overall_ndgs'
                overall_pos = 'overall_pos'

                df = (
                    predictions
                    .dataframe()
                    .select(
                        'ndg',
                        ref_dt,
                        target_col,
                        prediction_col,
                        F.percent_rank().over(Window.partitionBy('reference_date_m').orderBy(prediction_col)).alias(percent_rank_col),
                    )
                    .withColumn(temp_percentile_col, (F.col(percent_rank_col) * quantile_granularity).cast(IntegerType()))
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

                metrics.write_dataframe(metrics_df)

            transforms.append(compute_model_metrics)

    return transforms


TRAIN_TRANSFORMS = prediction_generator(exp_manager.config_experiment, exp_manager.BASELINE_MODELS)
