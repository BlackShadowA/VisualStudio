from cards_propensity_experiments.utils.const import FEATURES_FAMILIES, MUST_HAVE_FEATS
from mlops_experiments.experiment import ExperimentUtils
from hyperopt_broadcast_fix.pyll.base import scope
from hyperopt_broadcast_fix import hp
from typing import Dict, Any
import lightgbm as lgb
import catboost as cb
import numpy as np

EXPERIMENT_NAME = 'catboost_tuned'
BASE_PATH = "/uci/dp_marketing_models/technical/apps/Cards/Individuals/data/train"

TARGET_VARIABLE = 'target'

# ------------------------------------- FEATURE SELECTION

FEATURE_CLASSIFICATION_METHOD = ExperimentUtils(method_name="nulls_detector").get_feature_selection()
FEATURE_SELECTION_UNI_METHOD = ExperimentUtils(method_name="univariate_decorrelation").get_feature_selection()
FEATURE_SELECTION_MULTI_METHOD = ExperimentUtils(method_name="bagging_feature_importance").get_feature_selection()
FEATURE_SELECTION_MULTI_METHOD_2 = ExperimentUtils(method_name="quantile_performance").get_feature_selection()

FEATURE_CLASSIFICATION_PARAMS: Dict[str, Any] = {
    "classification_task": True,
    "clf_distinct_fl": True,
    "discrete_thr": 0.025,
    "min_distinct_values": 2,
    "null_perc": 0.95,
    "std_thr": 0.001,
    "thr_few_many_nulls": 0.75,
    "target_col": TARGET_VARIABLE
    }


FEATURE_SELECTION_UNI_PARAMS: Dict[str, Any] = {
    "classification_task": True,
    'family_feats': FEATURES_FAMILIES,
    "target_col": TARGET_VARIABLE,
    "corr_th": 0.7,
    "dendogram_thr_many": 0.1,
    "jaccard_thr_many": 0.05,
    "dendogram_thr_few": 0.15,
    "jaccard_thr_few": 0.1,
    "custom_feats": None,
    # "cols_to_drop": List[str] = None,
    # "bypass_feats": List[str] = None, #must have feats
    }

FEATURE_SELECTION_MULTI_PARAMS: Dict[str, Any] = {
    "model": lgb.LGBMClassifier(),
    "model_params": {
                    "max_depth": 3,
                    "colsample_bytree": 0.04,
                    "boosting_type": 'gbdt',
                    "subsample": 0.5,
                    "importance_type": 'gain',
                    "random_state": 0
            },
    "target_col": TARGET_VARIABLE,
    "df_partitions": 1000,
    "threshold": None,
    "bag_size_perc": 30,
    "n_bags": 100
    }

FEATURE_SELECTION_QUANTILE: Dict[str, Any] = {
    "model": lgb.LGBMClassifier(),
    "model_params": {
                    "n_estimators": 128,
                    "importance_type": 'gain',
                    "colsample_bytree": 0.04,
                    "subsample": .5,
                    "n_jobs": -1,
                    "learning_rate": 0.1,
                    "max_depth": 5
            },
    "target_col": "target",
    "df_partitions": 3000,
    "n_quantile": 50,
    "rank_col": 'rank',
    "quantile_sel_metric": 'aucpr',
    "quantile_sel_increasing": True,
    "quantile_sel_thr": 0.5,
    "aucpr_pos_label": 1,
    }


# ------------------------------------- MODEL

MODEL = cb.CatBoostClassifier()
HYPERPARAMS_TUNING = ExperimentUtils(method_name="hyperopt").get_hp_tuning()

# ------------------------------------- HYPER-PARAMTERE TUNING

HYPERPARAMS_TUNING_SEARCH_SPACE = {
    'n_estimators': 512,
    'learning_rate': hp.loguniform('cb_learning_rate', np.log(0.01), np.log(0.1)),
    # 'num_leaves': scope.int(hp.quniform('cb_num_leaves', 30, 128, 1)),  # Can be used only with the Lossguide growing policies.
    # 'min_data_in_leaf':  scope.int(hp.quniform('cb_min_data_in_leaf', 32, 512, 1)),  # Can be used only with the Lossguide and Depthwise growing policies.
    'max_depth': scope.int(hp.quniform('cb_max_depth', 4, 8, 1)),
    'subsample': hp.uniform('cb_subsample', 0.7, 1),  # only if bootstrap types is in Poisson, Bernoulli, MVS
    'reg_lambda': hp.uniform('cb_reg_lambda', 0, 100),
    'rsm': hp.uniform('cb_rsm', 0.7, 1),
    # to be insert param to balance data during training process: auto_class_weights or scale_pos_weights
    }

FIT_PARAMS = {
        # "eval_metric": "PRAUC",
        "early_stopping_rounds": 100,
        # "bootstrap_type": "MVS"
    }

################################################################## EXPERIMENT DICIONARY ##################################################################

config_experiment: Dict[str, Any] = {
        'standard': {
            "name": EXPERIMENT_NAME,
            "train": f"{BASE_PATH}/train_test",
            "experiments_location": f"{BASE_PATH}/experiments",
            "output_location": f"{BASE_PATH}/model",
            'feat_selection': [
                FEATURE_CLASSIFICATION_PARAMS,
                FEATURE_SELECTION_UNI_PARAMS,
                FEATURE_SELECTION_MULTI_PARAMS,
                FEATURE_SELECTION_QUANTILE
            ],
            "must_have_feats": MUST_HAVE_FEATS
            },
    }