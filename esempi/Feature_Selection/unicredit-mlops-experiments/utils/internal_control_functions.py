import pyspark.sql
from private_const import COLS_TO_DROP
import pandas as pd  # noqa
from typing import Union, List
from sklearn.base import BaseEstimator
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, SparkSession


def _ipt_param_check(method_name: str):
    if not isinstance(method_name, str) or method_name == '':
        raise ValueError('method_name must be a not empty string')
    else:
        # manage wrong caps chars
        method_name = method_name.lower()
        if method_name not in ["hyperopt", "passthrough_hp_tuning",
                               "feature_importance", "multi_feature_importance",
                               "bagging_feature_importance", "multi_bagging_feature_importance",
                               "quantile_performance", "multi_quantile_performance",
                               "nulls_detector", "feature_classification", "uni_feature_classification",
                               "univariate_decorrelation", "uni_feature_decorrelation",
                               "univariate_best_feats", "uni_feature_related",
                               "passthrough_feature_selection", "uni_target_score", "multi_somers_delta_score",
                               "uni_correlation_matrix_score"]:
            raise ValueError('''method_name must be one of the implemented methods "hyperopt", "passthrough_hp_tuning",
                                "feature_importance or multi_feature_importance",
                                "bagging_feature_importance or multi_bagging_feature_importance",
                                "quantile_performance or multi_quantile_performance",
                                "nulls_detector or feature_classification or uni_feature_classification",
                                "univariate_decorrelation or uni_feature_decorrelation",
                                "univariate_best_feats or uni_feature_related", "passthrough_feature_selection",
                                "uni_target_score", "multi_somers_delta_score", "uni_correlation_matrix_score"''')


def _ipt_features_checkdrop(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:

    wrong_type_cols = [col[0] for col in df.dtypes if col[1] in ['timestamp', 'date', 'string']]

    return df.drop(*wrong_type_cols)


def _convert_pandas_to_pyspark_df(spark_session: pyspark.sql.SparkSession,
                                  df: Union[pyspark.sql.DataFrame, pd.DataFrame],
                                  custom_cols_to_drop: List[str],
                                  key_cols: List[str] = None,
                                  ) -> pyspark.sql.DataFrame:
    """checking the input dataframe and managing the conversation to the right type
    dropping not needed default columns

        Parameters
        ----------
        spark_session : pyspark.sql.SparkSession
           SparkSession from pyspark.sql
        df : Union[pyspark.sql.DataFrame, pd.DataFrame]
           An input dataframe (Pyspark or Pandas) with only the train features (no IDs, dates, additional informations, ...)
        custom_cols_to_drop: List[str]
           list of columns chosen by the user to drop before feature selection computation (e.g. IDs like ndg or cntp_id, reference dates like snapshot_date or reference_date_m and additional cols like in_sample, timestamp, etc)
        key_cols: List[str]
            list of columns that i want to save from the automatic drop (not cols_to_drop)

        Returns
        -------
        pyspark.sql.DataFrame
            Returns a pandas dataframe object

        Raises
        ------
        NotImplementedError
            Whenever the input df is neither a PySpark DataFrame, nor a Pandas DataFrame
        """
    tot_cols = COLS_TO_DROP
    custom_cols_to_drop = custom_cols_to_drop or []
    key_cols = key_cols or []
    if isinstance(custom_cols_to_drop, list):
        if len(custom_cols_to_drop) > 0:
            tot_cols = tot_cols + custom_cols_to_drop
    if isinstance(key_cols, list):
        if len(key_cols) > 0:
            tot_cols = [i for i in tot_cols if i not in key_cols]

    if isinstance(df, pd.DataFrame):
        df = spark_session.createDataFrame(df).drop(*tot_cols)
    elif isinstance(df, pyspark.sql.DataFrame):
        df = df.drop(*tot_cols)
    else:
        raise NotImplementedError(
            f"Input dataset is of type {type(df)}: the method support only Pandas or PySpark DataFrames as input datasets"
        )
    return df


def _convert_pyspark_to_pandas_df(spark_session: pyspark.sql.SparkSession,
                                  df:  Union[pyspark.sql.DataFrame, pd.DataFrame],
                                  custom_cols_to_drop: List[str],
                                  key_cols: List[str] = None,
                                  ) -> pd.DataFrame:
    """checking the input dataframe and managing the conversation to the right type
    dropping not needed default columns

        Parameters
        ----------
        spark_session : pyspark.sql.SparkSession
           SparkSession from pyspark.sql
        df : Union[pyspark.sql.DataFrame, pd.DataFrame]
           An input dataframe (Pyspark or Pandas) with only the train features (no IDs, dates, additional informations, ...)
        custom_cols_to_drop: List[str]
           list of columns chosen by the user to drop before feature selection computation (e.g. IDs like ndg or cntp_id, reference dates like snapshot_date or reference_date_m and additional cols like in_sample, timestamp, etc)
        key_cols: List[str]
            list of columns that i want to save from the automatic drop (not cols_to_drop)


        Returns
        -------
        pd.DataFrame
            Returns a pandas dataframe object

        Raises
        ------
        NotImplementedError
            Whenever the input df is neither a PySpark DataFrame, nor a Pandas DataFrame
        """
    tot_cols = COLS_TO_DROP
    custom_cols_to_drop = custom_cols_to_drop or []
    key_cols = key_cols or []
    if isinstance(custom_cols_to_drop, list):
        if len(custom_cols_to_drop) > 0:
            tot_cols = tot_cols + custom_cols_to_drop
    if isinstance(key_cols, list):
        if len(key_cols) > 0:
            tot_cols = [i for i in tot_cols if i not in key_cols]

    if isinstance(df, pyspark.sql.DataFrame):
        df = (
            df
            .drop(*tot_cols)
            .toPandas()
        )
    elif isinstance(df, pd.DataFrame):
        df = (
            df
            .drop(tot_cols, axis=1, errors='ignore')
        )
    else:
        raise NotImplementedError(
            f"Input dataset is of type {type(df)}: the method support only Pandas or PySpark DataFrames as input datasets"
        )
    return df


def _conditional_predict(X: pd.DataFrame, model: BaseEstimator, classification_task: bool):
    """checking the objective and perform a differenct predict method depends on the task

        Parameters
        ----------
        X: pd.DataFrame
           input with features
        model: BaseEstimator
           model used to estimate
        classification_task: bool
           flag to set the objective

        Returns
        -------
        pd.DataFrame
            Returns a pandas dataframe object
        """

    return model.predict_proba(X)[:, 1].squeeze() if classification_task else model.predict(X).squeeze()


def _sparkmelt(
        df: DataFrame, id_cols: List[str], value_vars: List[str] = [],
        var_name: str = 'variable', value_name: str = 'value'
        ) -> DataFrame:
    '''
    PySpark version of the Pandas Melt function.
    - Parameters:
            `df`: dataframe to melt.
            `id_cols`: columns to use as identifier variables.
            `value_vars`: columns to unpivot, defaults to all non id_cols columns.
            `var_name`: name to use for the variable column.
            `value_name`: name to use for the column containing the values.

    - Returns:
            A DataFrame melted like for the Pandas `melt` function, see the following link:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html
    '''

    # Create an array of structs for each value_vars
    value_vars = value_vars if value_vars else df.drop(*id_cols).columns
    tmp = F.array(*(F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name)) for c in value_vars))

    # Add it to the DataFrame and explode
    df = df \
        .withColumn("tmp", F.explode(tmp)) \
        .select(*id_cols, *[F.col("tmp")[x].alias(x) for x in (var_name, value_name)])

    return df


def _impose_order(df: DataFrame, key_cols: List[str], n_partitions: int = 200):
    '''
    Impose a total reproducible order on the DataFrame to make the subsequent operations deterministic.
    - Parameters:
            `df`: input dataframe.
            `key_cols`: list of columns to create a unique key for the pair RDD conversion.
            `n_partitions`: number of partitions.
    - Returns:
            A DataFrame with `n_partitions` partitions ordered by `key`.
    '''

    # Handle a single column passed as string
    key_cols = key_cols if isinstance(key_cols, (list, tuple)) else [key_cols]

    # Convert to pair RDD and sort
    rdd_sorted = (
        df
        .rdd
        .keyBy(lambda t: tuple(t[c] for c in key_cols))
        .repartitionAndSortWithinPartitions(n_partitions)
        .values()
    )

    # Convert back to DataFrame
    spark = SparkSession.builder.getOrCreate()
    df_sorted = spark.createDataFrame(rdd_sorted, schema=df.schema)

    return df_sorted