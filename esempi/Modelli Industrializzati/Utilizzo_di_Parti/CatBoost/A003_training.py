from transforms.api import transform, Input, Output, configure
import cards_propensity_experiments.utils.experiment_manager as exp_manager
# from cards_propensity_experiments.utils.functions import AUPRC_lgbm
# import lightgbm as lgb
from hyperopt_broadcast_fix import STATUS_OK
from foundry_ml_utils.model import ModelOps
from pyspark.sql import functions as F
import pandas as pd


# Hyperopt hyperparameters
EI_CANDIDATES = 48
MAX_OPT_ITER = 64
N_RANDOM_STARTS = 32
PARALLELISM = 32
OBJECTIVE = 'Logloss'

# catboost hyperparameters
ES_ROUNDS = 64
# N_ESTIMATORS = 512
# N_JOBS = -1
# Other hyperparameters
SEED = 42
# EVAL_METRIC = AUPRC_lgbm

static_model_params = {
    # "n_estimators": N_ESTIMATORS,
    'loss_function': OBJECTIVE,
    "eval_metric": 'PRAUC',
    "bootstrap_type": "MVS",
    # "n_jobs": -1,
    "random_state": SEED
    }


def train_generator(config_dict):

    transforms = []

    for _, experiment in config_dict.items():

        @configure(profile=[
            'EXECUTOR_MEMORY_LARGE',
            'DRIVER_MEMORY_EXTRA_LARGE',
            'DRIVER_MEMORY_OVERHEAD_LARGE',
            'EXECUTOR_MEMORY_OVERHEAD_LARGE'
        ])
        @transform(
            out_model=Output(f"{experiment['output_location']}/model"),
            out_model_metadata=Output(f"{experiment['experiments_location']}/model_metadata"),
            feature_selection=Input(f"{experiment['experiments_location']}/feat_selection_multivariate_quantile_performance"),
            all_feats_val=Input(f"{experiment['train']}/train_val_retail_with_feats", branch="master"),
            all_feats_train=Input(f"{experiment['train']}/train_train_retail_with_feats", branch="master"),
        )
        def compute_optimal_model(
            ctx,
            out_model,
            out_model_metadata,
            feature_selection,
            all_feats_train,
            all_feats_val,
            config=experiment
        ):

            best_quantile = (
                feature_selection
                .dataframe()
                .select('best_quantile')
                .rdd
                .flatMap(lambda x: x)
                .collect()[0]
            )

            best_features = (
                feature_selection
                .dataframe()
                .filter(F.col('ts') == 'test')
                .filter(F.col('run') == best_quantile)
                .filter(F.col('performance') == 'aucpr')
                .select('feature_name')
                .rdd
                .flatMap(lambda x: x[0])
                .collect()
            )

            train_df = (
                all_feats_train
                .dataframe()
                .select('ndg', 'reference_date_m', exp_manager.TARGET_VARIABLE, *best_features)
            ).toPandas()

            val_df = (
                all_feats_val
                .dataframe()
                .select('ndg', 'reference_date_m', exp_manager.TARGET_VARIABLE, *best_features)
            ).toPandas()

            X_train = ctx.spark_session.sparkContext.broadcast(
                train_df[best_features].astype(float)
            )
            y_train = ctx.spark_session.sparkContext.broadcast(
                train_df[exp_manager.TARGET_VARIABLE].values
            )
            X_val = ctx.spark_session.sparkContext.broadcast(
                val_df[best_features].astype(float)
            )
            y_val = ctx.spark_session.sparkContext.broadcast(
                val_df[exp_manager.TARGET_VARIABLE].values
            )

            hyperparams_tuning = exp_manager.HYPERPARAMS_TUNING

            hyperparams_tuning_params = {
                    "ei_candidate": EI_CANDIDATES,
                    "max_opt_iter": MAX_OPT_ITER,  # change to 50
                    "n_random_starts": N_RANDOM_STARTS,
                    "spark_trials_fl": True,
                    "parallelism": PARALLELISM,
                    "seed": SEED,
                    "search_space": exp_manager.HYPERPARAMS_TUNING_SEARCH_SPACE,
                    "verbose": False,
                    "num_boosting_rounds_fl": False,
                }
            hyperparams_tuning.set_params(**hyperparams_tuning_params)
            # set the model
            model = exp_manager.MODEL
            objective_params = {
                "X_train_": X_train,
                "y_train_": y_train,
                "X_val_": X_val,
                "y_val_": y_val,
                "model": model,
                }

            all_trials, best_params, num_boosting_rounds = hyperparams_tuning.compute(
                objective_func=_objective,
                objective_params=objective_params
                )

            all_params = {
                **static_model_params,
                **best_params
            }

            model_metadata = pd.DataFrame([all_params])
            out_model_metadata.write_dataframe(ctx.spark_session.createDataFrame(model_metadata))

            X_val = val_df[best_features].astype(float)  # .values
            y_val = val_df[exp_manager.TARGET_VARIABLE].values
            fit_params = exp_manager.FIT_PARAMS
            fit_params['eval_set'] = [(X_val, y_val)]

            model = exp_manager.MODEL
            model.set_params(**all_params)
            model_to_serialize = ModelOps.create_model_pipeline(
                training_df=train_df,
                target_col=exp_manager.TARGET_VARIABLE,
                model=model,
                features=best_features,
                fit_args=fit_params
            )
            model_to_serialize.save(out_model)
        transforms.append(compute_optimal_model)
    return transforms


TRAIN_TRANSFORMS = train_generator(exp_manager.config_experiment)


# --------------------------------------- INTERNAL ----------------------------------------------


def _objective(params, X_train_, y_train_, X_val_, y_val_, model):

    X_train = X_train_.value
    y_train = y_train_.value
    X_val = X_val_.value
    y_val = y_val_.value

    model_params = {
        # 'n_estimators': N_ESTIMATORS,
        'loss_function': OBJECTIVE,
        "eval_metric": 'PRAUC',
        "bootstrap_type": "MVS",
        # 'early_stopping_rounds': ES_ROUNDS,
        'random_state': SEED
    }
    obj_params = {
                **model_params,
                **params
            }
    model.set_params(**obj_params)
    # model = lgb.LGBMClassifier(
    #     n_estimators=N_ESTIMATORS,
    #     objective=OBJECTIVE,
    #     n_jobs=-1,
    #     metric='custom',
    #     random_state=SEED,
    #     **params
    # )

    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=ES_ROUNDS,
        # eval_metric='PRAUC',
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # y_val_preds = model.predict_proba(X_val)[:, 1]

    # from sklearn.metrics import auc, precision_recall_curve

    # precision, recall, _ = precision_recall_curve(y_val, y_val_preds, pos_label=1)

    # val_auprc = auc(recall, precision)

    # return {
    #     'loss': -val_auprc,
    #     'status': STATUS_OK
    # }
    return {
        'loss': model.get_best_score()['validation']['PRAUC'],
        'status': STATUS_OK
        }
