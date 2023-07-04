from typing import Callable, List, Tuple

import catboost as cb
import lightgbm as lgb
import pandas as pd  # noqa
import xgboost as xgb
from foundry_ml_utils.model import ModelOps
from foundry_ml import Model
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from cards_propensity_experiments.utils.functions import AUPRC_lgbm


class BaselineModels:

    metric: Callable = AUPRC_lgbm
    target_col: str = "target"
    random_state: int = 7
    early_stopping_rounds: int = 100
    n_jobs: int = -1

    @staticmethod
    def _fill_na(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fill_with_zero = [
            c for c in X_train.columns if ("sum" in c) | ("cnt" in c) | ("count" in c)
        ]
        X_train[fill_with_zero] = X_train[fill_with_zero].fillna(0)
        X_train = X_train.fillna(-9999)
        X_val[fill_with_zero] = X_val[fill_with_zero].fillna(0)
        X_val = X_val.fillna(-9999)
        return X_train, X_val

    @staticmethod
    def get_lgb(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = lgb.LGBMClassifier(
            n_estimators=512,
            objective="binary",
            max_depth=5,
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric=BaselineModels.metric,
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_lgb_balanced(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = lgb.LGBMClassifier(
            n_estimators=512,
            objective="binary",
            max_depth=5,
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
            is_unbalance=True,
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric=BaselineModels.metric,
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_isolation_forest(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        X_train, X_val = BaselineModels._fill_na(X_train=X_train, X_val=X_val)
        clf = IsolationForest(
            n_estimators=256,
            max_samples=0.4,
            contamination=0.001,
            max_features=0.7,
            bootstrap=True,
            n_jobs=BaselineModels.n_jobs,
            random_state=BaselineModels.random_state,
        )

        # TODO: sovrascrivere transform per fare score_samples

        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
        )

    @staticmethod
    def get_lgb_goss(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = lgb.LGBMClassifier(
            n_estimators=512,
            objective="binary",
            max_depth=5,
            boosting_type="goss",
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric=BaselineModels.metric,
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_lgb_goss_balanced(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = lgb.LGBMClassifier(
            n_estimators=512,
            objective="binary",
            max_depth=5,
            boosting_type="goss",
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
            is_unbalance=True,
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric=BaselineModels.metric,
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_xgb_gb_rf_balanced(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = xgb.XGBClassifier(
            n_estimators=512,
            max_depth=5,
            scale_pos_weight=5,
            sampling_method="gradient_based",
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
            booster="gbtree",
            num_parallel_tree=16,
        )
        #X_val = X_val#.values
        #y_val = y_val#.values
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric="aucpr",
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_xgb_gb(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = xgb.XGBClassifier(
            n_estimators=512,
            max_depth=5,
            sampling_method="gradient_based",
            num_parallel_tree=16,
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
            booster="gbtree",
        )
        #X_val = X_val#.values
        #y_val = y_val#.values
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric="aucpr",
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_xgb_balanced_rf(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = xgb.XGBClassifier(
            n_estimators=128,
            max_depth=5,
            num_parallel_tree=32,
            scale_pos_weight=5,
            n_jobs=BaselineModels.n_jobs,
            metric="custom",
            random_state=BaselineModels.random_state,
            booster="gbtree",
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_metric="aucpr",
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_svc(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        X_train, X_val = BaselineModels._fill_na(X_train=X_train, X_val=X_val)
        clf = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=True,
            tol=0.001,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=1024,
            decision_function_shape="ovr",
            random_state=BaselineModels.random_state,
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
        )

    @staticmethod
    def get_cb(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = cb.CatBoostClassifier(
            n_estimators=256,
            max_depth=5,
            eval_metric="PRAUC",
            random_seed=BaselineModels.random_state,
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_cb_mvs_balanced(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = cb.CatBoostClassifier(
            n_estimators=256,
            max_depth=5,
            scale_pos_weight=5,
            eval_metric="PRAUC",
            bootstrap_type="MVS",
            random_seed=BaselineModels.random_state,
        )
        fit_args = dict(
            early_stopping_rounds=BaselineModels.early_stopping_rounds,
            eval_set=[(X_val, y_val)],
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train,
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            fit_args=fit_args,
        )

    @staticmethod
    def get_random_forest(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
    ) -> Model:
        clf = RandomForestClassifier(
            n_estimators=256,
            max_depth=5,
            criterion='gini',
            random_state=BaselineModels.random_state,
        )
        return ModelOps.create_model_pipeline(
            training_df=X_train.fillna(0),
            target_col=BaselineModels.target_col,
            model=clf,
            features=features,
            #fit_args=fit_args,
        )