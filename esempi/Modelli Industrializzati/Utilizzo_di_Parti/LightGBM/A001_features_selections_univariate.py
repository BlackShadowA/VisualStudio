from transforms.api import transform, Input, Output, configure
import cards_propensity_experiments.utils.experiment_manager as exp_manager
import pandas as pd


def transform_generator(config_dict):

    transforms = []

    for _, experiment in config_dict.items():
        @configure(profile=[
            'DRIVER_MEMORY_EXTRA_LARGE',
            'EXECUTOR_MEMORY_LARGE'
        ])
        @transform(
            feat_univ=Output(f"{experiment['experiments_location']}/feat_selection_univariate"),
            feat_class=Output(f"{experiment['experiments_location']}/feat_classification"),
            train_all_feats=Input(f"{experiment['train']}/train_train_retail_with_feats", branch="master"),
        )
        def compute_fs(
            feat_class,
            feat_univ,
            train_all_feats,
            ctx,
            config=experiment
        ):
            # remove 
            train_all_feats = train_all_feats.dataframe().drop('set', 'F070010_card__credit_has_atleast_one_contactless')
            categorizer_feats = [s for s in train_all_feats.columns if 'categorizer_' in s]
            train_all_feats = train_all_feats.drop(*categorizer_feats)

            feature_classification = exp_manager.FEATURE_CLASSIFICATION_METHOD
            feature_classification.set_params(**config['feat_selection'][0])

            feature_selection = exp_manager.FEATURE_SELECTION_UNI_METHOD
            feature_selection.set_params(**config['feat_selection'][1])

            feats_classifed = feature_classification.compute(
                ctx.spark_session,
                train_all_feats
            )
            feats_univariate = feature_selection.compute(
                ctx.spark_session,
                df_train=train_all_feats,
                df_with_nulls=feats_classifed
            )

            feats_univariate_list = list(feats_univariate.select('feature_name').toPandas()['feature_name'])

            must_have_feats = [
                str(c) for c in config['must_have_feats'] if (c in train_all_feats.columns) and (c not in feats_univariate_list)
            ]

            feat_final = ctx.spark_session.createDataFrame(
                pd.DataFrame({'feature_name': must_have_feats + feats_univariate_list})
            )

            feat_class.write_dataframe(feats_classifed)
            feat_univ.write_dataframe(feat_final)

        transforms.append(compute_fs)
    return transforms


TRAIN_TRANSFORMS = transform_generator(exp_manager.config_experiment)

