from mlops_experments.feature_selection.fs_univariate.univariate_feature_related.feature_classification import UniFeatureClassification
from utils.internal_control_functions import _ipt_param_check

# ------- EXPERIMENT UTILS: Configuration file to set up all MLOps experiments ------ #


class ExperimentUtils(object):
    """Overall class to import all methods implemented"""

    def __init__(self, method_name: str) -> None:
        """

        Parameters
        ----------
        method_name: str
           name of one of the implemented methods
        Raises
        ------
        ValueError
            Whenever 'method_name' is set up with wrong data type
        """

        # check the input
        _ipt_param_check(method_name)

        # manage wrong caps chars
        method_name = method_name.lower()

        self.method_name = method_name

    '''
        FEATURE SELECTION UTILS

        Private vars to set up the chosen feature selection method and params for the experiment

    '''

    # dictionary with the implemented feature selection methods
    _feature_selection_methods = {
        #"feature_importance": MultiFeatureImportance(),  # keeped for backward compatibility
        #"multi_feature_importance": MultiFeatureImportance(),
        #"bagging_feature_importance": MultiBaggingFeatureImportance(),  # keeped for backward compatibility
        #"multi_bagging_feature_importance": MultiBaggingFeatureImportance(),
        #"quantile_performance": MultiQuantilePerformance(),  # keeped for backward compatibility
        #"multi_quantile_performance": MultiQuantilePerformance(),
        "nulls_detector": UniFeatureClassification(),  # keeped for backward compatibility
        #"feature_classification": UniFeatureClassification(),  # keeped for backward compatibility (2)
        #"uni_feature_classification": UniFeatureClassification(),
        #"univariate_decorrelation": UniFeatureDecorrelation(),  # keeped for backward compatibility
        #"uni_feature_decorrelation": UniFeatureDecorrelation(),
        #"univariate_best_feats": UniFeatureRelated(),  # keeped for backward compatibility
        #"uni_feature_related": UniFeatureRelated(),
        #"passthrough_feature_selection": PassthroughFeatureSelection(),
        #"uni_target_score": UniTargetRelatedScore(),
        #"multi_somers_delta_score": MultiSomersDeltaScore(),
        #"uni_correlation_matrix_score": UniCorrelationMatrixScore()

    }

    # FEATURE SELECTION SERVING METHOD

    def get_feature_selection(self) -> object:
        """Serving all implemented feature selection methods

        Returns
        -------
        object:
            Returns the feature selection method object requested with default params setting.
            Use get_feature_selection().set_params(**custom_params) to set one or more new custom params for the chosen object

        Raises
        ------
        ValueError
            Whenever 'method_name' it is not one of the implemented methods
        """

        if self.method_name not in ["feature_importance", "multi_feature_importance",
                                    "bagging_feature_importance", "multi_bagging_feature_importance",
                                    "quantile_performance", "multi_quantile_performance",
                                    "nulls_detector", "feature_classification", "uni_feature_classification",
                                    "univariate_decorrelation", "uni_feature_decorrelation",
                                    "univariate_best_feats", "uni_feature_related",
                                    "passthrough_feature_selection", "uni_target_score", "multi_somers_delta_score",
                                    "uni_correlation_matrix_score"]:

            raise ValueError('''method_name must be one of the implemented methods "feature_importance or multi_feature_importance",
                             "bagging_feature_importance or multi_bagging_feature_importance",
                             "quantile_performance or multi_quantile_performance",
                             "nulls_detector or feature_classification or uni_feature_classification",
                             "univariate_decorrelation or uni_feature_decorrelation",
                             "univariate_best_feats or uni_feature_related", "passthrough_feature_selection",
                             "uni_target_score", "multi_somers_delta_score", "uni_correlation_matrix_score"''')

        return ExperimentUtils._feature_selection_methods[self.method_name]

    '''
        HYPERPARAMETER TUNING UTILS

        Private dict to set up the chosen hyperparameter tuning method and params for the experiment

    
    
    # dictionary with the implemented hyperparameters tuning methods
    _hp_tuning_methods = {
        "hyperopt": Hyperopt(),
        "passthrough_hp_tuning": PassthroughHpTuning()

    }

    def get_hp_tuning(self) -> object:
        """Serving all implemented hyeperparameters tuning methods

        Returns
        -------
        object:
            Returns the hyeperparameters tuning method object requested with default params setting.
            Use get_hp_tuning().set_params(**custom_params) to set one or more new custom params for the chosen object

        Raises
        ------
        ValueError
            Whenever 'method_name' it is not one of the implemented methods
        """
        if self.method_name not in ["hyperopt", "passthrough_hp_tuning"]:
            raise ValueError('method_name must be one of the implemented methods "hyperopt", "passthrough_hp_tuning')

        return ExperimentUtils._hp_tuning_methods[self.method_name]
    '''