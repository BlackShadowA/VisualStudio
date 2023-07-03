from transforms.api import transform, Input, Output, configure
import cards_propensity_experiments.utils.experiment_manager as exp_manager


def transform_generator(config_dict):

    transforms = []

    for _, experiment in config_dict.items():

        @configure(profile=[
            'DRIVER_MEMORY_EXTRA_LARGE',
            'DRIVER_CORES_MEDIUM',
            'DRIVER_MEMORY_OVERHEAD_LARGE',
            'EXECUTOR_MEMORY_LARGE',
            'EXECUTOR_MEMORY_OVERHEAD_EXTRA_LARGE',
            'SHUFFLE_PARTITIONS_LARGE',
            'NUM_EXECUTORS_64'
        ])
        @transform(
            quantile_ranking=Output(f"{experiment['experiments_location']}/feat_selection_multivariate_quantile_performance"),
            feature_importance=Output(f"{experiment['experiments_location']}/feat_selection_multivariate_importance"),
            train_all_feats=Input(f"{experiment['train']}/train_train_retail_with_feats", branch="master"),
            valid_all_feats=Input(f"{experiment['train']}/train_val_retail_with_feats", branch="master"),
            uni_df=Input(f"{experiment['experiments_location']}/feat_selection_univariate")
        )
        def compute_fs(
            quantile_ranking,
            feature_importance,
            train_all_feats,
            valid_all_feats,
            uni_df,
            ctx,
            config=experiment
        ):

            feat_selected_univariate = list(uni_df.dataframe().select('feature_name').toPandas()['feature_name'])
            feat_selected_univariate.append('target')

            train_selected = (
                train_all_feats
                .dataframe()
                .select(feat_selected_univariate)
            )
            val_selected = (
                valid_all_feats
                .dataframe()
                .select(feat_selected_univariate)
            )

            feature_importance_selection = exp_manager.FEATURE_SELECTION_MULTI_METHOD
            feature_importance_selection.set_params(**config['feat_selection'][2])

            quantile_performance_selection = exp_manager.FEATURE_SELECTION_MULTI_METHOD_2
            quantile_performance_selection.set_params(**config['feat_selection'][3])

            feats_multi = feature_importance_selection.compute(
                ctx.spark_session,
                train_selected
            )

            feats_multi_2 = quantile_performance_selection.compute(
                ctx.spark_session,
                train_selected,
                val_selected,
                feats_multi
            )

            feature_importance.write_dataframe(feats_multi)
            quantile_ranking.write_dataframe(feats_multi_2)

        transforms.append(compute_fs)
    return transforms


TRAIN_TRANSFORMS = transform_generator(exp_manager.config_experiment)

