from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
import cards_propensity_experiments.utils.experiment_manager as exp_manager
from cards_propensity_experiments.utils.functions import compute_predictions


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
                predictions=Output(f"{experiment['output_location']}/inference/{exp_manager.EXPERIMENT_NAME}_{model_name}_predictions"),
                model=Input(f"{experiment['output_location']}/{exp_manager.EXPERIMENT_NAME}_{model_name}", branch="baseline_models"),
                all_feats_test=Input(f"{experiment['train']}/test_retail_with_feats", branch="master")
            )
            def compute_model_predictions(
                ctx,
                predictions,
                model,
                all_feats_test,
                model_dict=model_exp
            ):

                feats = exp_manager.FEATURE_SELECTION_BASELINE

                test_df = (
                    all_feats_test
                    .dataframe()
                    .select(
                        'ndg',
                        'reference_date_m',
                        exp_manager.TARGET_VARIABLE,
                        *[F.col(feat).cast('float').alias(feat) for feat in feats]
                    )
                )

                preds = compute_predictions(model_obj=model, features=feats, test=test_df, mod_name=model_name, n_partitions=2000)
                predictions.write_dataframe(preds)

            transforms.append(compute_model_predictions)

    return transforms


TRAIN_TRANSFORMS = prediction_generator(exp_manager.config_experiment, exp_manager.BASELINE_MODELS)
