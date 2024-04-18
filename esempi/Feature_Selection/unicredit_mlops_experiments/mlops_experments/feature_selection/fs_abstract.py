from abc import abstractmethod
from typing import Any, Dict, Union, List
import pyspark.sql
# from sklearn.base import BaseEstimator
import pandas as pd  # noqa
from unicredit_mlops_experiments import root
from unicredit_mlops_experiments.root import BaseClass


# ------- FEATURE SELECTION ------ #

class FeatureSelection(BaseClass):
    """Abstract class to reflect the implementation of a feature selection method"""

    name: str = 'Feature Selection'
    target_col: str
    cols_to_drop: List[str]

    def __init__(
        self,
        target_col: str,
        cols_to_drop: List[str],
        **kwargs: Dict[str, Any]
    ) -> None:
        """

        Parameters
        ----------
        target_col: str
           name of the target/label column of the training dataframe
        cols_to_drop: List[str]
           list of columns to drop before feature selection computation (e.g. IDs like ndg or cntp_id, reference dates like snapshot_date or reference_date_m and additional cols like in_sample, timestamp, etc)
        **kwargs : Dict[str, Any]
            Additional keyword arguments
        """
        self.name = FeatureSelection.name
        self.target_col = target_col
        self.cols_to_drop = cols_to_drop
        self._other_params: Dict[str, Any] = {}
        self.set_params(**kwargs)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this object.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this object and
            contained subobjects that are objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super()._get_params(deep=deep)
        params.update(self._other_params)
        return params

    def set_params(self, **params: Any):
        """Set the parameters of this object.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
        return super()._set_params(**params)

    @abstractmethod
    def compute(
        self,  spark_session: pyspark.sql.SparkSession,
        df_train: Union[pyspark.sql.DataFrame, pd.DataFrame],
        **kwargs
    ) -> pyspark.sql.DataFrame:
        """Abstract method that each feature selection will need to implement

        Parameters
        ----------
        spark_session : pyspark.sql.SparkSession
           SparkSession from pyspark.sql
        df_train : Union[pyspark.sql.DataFrame, pd.DataFrame]
           An input dataframe (Pyspark or Pandas) with only the train features (no IDs, dates, additional informations, ...)
        **kwargs : Dict[str, Any]
            Additional keyword arguments

        Returns
        -------
        pyspark.sql.DataFrame
            Returns a pyspark dataframe object
        """
        pass


# # ------- MULTIVARIATE FEATURE SELECTION ------ #

# class MultivariateFeatureSelection(FeatureSelection):
#     """Class to reflect the implementation of a multivariate feature selection method"""

#     name: str = 'Multivariate Feature Selection'
#     model: BaseEstimator
#     model_params: Dict[str, Any]
#     fit_params: Dict[str, Any]
#     df_partitions: int

#     def __init__(
#         self,
#         model: BaseEstimator,
#         model_params: Dict[str, Any],
#         fit_params: Dict[str, Any],
#         target_col: str,
#         cols_to_drop: List[str],
#         df_partitions: int,
#     ) -> None:
#         """

#         Parameters
#         ----------
#         model : BaseEstimator
#            scikit-learn estimator
#         model_params: Dict[str, Any]
#            custom hyperparameters to set up the estimator, use None or {} to use deafult model parameters
#         fit_params: Dict[str, Any]
#            custom fit parameters to set up the training
#         target_col: str
#            name of the target/label column of the training dataframe
#         cols_to_drop: List[str]
#            list of columns to drop before feature selection computation (e.g. IDs like ndg or cntp_id, reference dates like snapshot_date or reference_date_m and additional cols like in_sample, timestamp, etc)
#         df_partitions: int
#            number of partitions for the input dataframe repartition
#         """
#         super(MultivariateFeatureSelection, self).__init__(target_col=target_col, cols_to_drop=cols_to_drop)
#         self.name = MultivariateFeatureSelection.name
#         self.model = model
#         self.model_params = model_params
#         self.fit_params = fit_params
#         self.df_partitions = df_partitions