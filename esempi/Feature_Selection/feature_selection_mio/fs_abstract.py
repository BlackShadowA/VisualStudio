from abc import abstractmethod
from typing import Any, Dict, Union, List
import pyspark.sql
# from sklearn.base import BaseEstimator
import pandas as pd  # noqa
from feature_selection_mio.classe_astract import BaseClass
'''
# from feature_selection_mio.classe_astratta import BaseClass
from abc import ABC
from typing import Any, Dict
import inspect

# ------- Classe Astratta ------ #
class BaseClass(ABC):
    """Abstract root class to reflect the implementation of a all methods"""

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "methods should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def _get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this method.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this method and
            contained subobjects that are methods.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value

        return out

    def _set_params(self, **params: Any):
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
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            self._other_params[key] = value
        return self
'''
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
