from transforms.api import transform, Input, Output, configure
import cards_propensity_experiments.utils.experiment_manager as exp_manager


def train_generator(config_dict, baselines):

    transforms = []

    for _, experiment in config_dict.items():

        for model_name in baselines:

            model_exp = baselines[model_name]

            @configure(profile=[
                'DRIVER_MEMORY_EXTRA_LARGE', 'AUTO_BROADCAST_JOIN_DISABLED', 'EXECUTOR_MEMORY_MEDIUM', 'NUM_EXECUTORS_16', 'DRIVER_MEMORY_OVERHEAD_LARGE'])
            @transform(
                out_model=Output(f"{experiment['output_location']}/{exp_manager.EXPERIMENT_NAME}_{model_name}"),
                all_feats_val=Input(f"{experiment['train']}/train_train_retail_with_feats",  branch="master"),
                all_feats_train=Input(f"{experiment['train']}/train_val_retail_with_feats", branch="master")
            )
            def compute_optimal_model(
                ctx,
                out_model,
                all_feats_train,
                all_feats_val,
                model_dict=model_exp
            ):

                features = exp_manager.FEATURE_SELECTION_BASELINE

                train_df = (
                    all_feats_train
                    .dataframe()
                    .select(exp_manager.TARGET_VARIABLE, *features)
                ).toPandas()

                val_df = (
                    all_feats_val
                    .dataframe()
                    .select(exp_manager.TARGET_VARIABLE, *features)
                ).toPandas()

                features_to_cast = {col: float for col in features}
                X_train = train_df.astype(features_to_cast)
                X_val = val_df[features].astype(float)
                y_val = val_df[exp_manager.TARGET_VARIABLE]

                model_to_serialize = model_dict(X_train=X_train, X_val=X_val, y_val=y_val, features=features)
                model_to_serialize.save(out_model)

            transforms.append(compute_optimal_model)

    return transforms


TRAIN_TRANSFORMS = train_generator(exp_manager.config_experiment, exp_manager.BASELINE_MODELS)
