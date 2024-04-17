from pyspark.sql import functions as F
from mlops_experiments.feature_selection.fs_abstract import FeatureSelection
from typing import List, Union
import pyspark.sql
import pandas as pd  # noqa
from mlops_experiments.utils.internal_control_functions import _convert_pandas_to_pyspark_df

# -------  FEATURE CLASSIFICATION ------ #


class UniFeatureClassification(FeatureSelection):
    """Class to classify features dividing into various sets: continuous - few nulls, continuous - many nulls, discrete - few nulls, discrete - many nulls"""

    name: str = 'feature classification'

    def __init__(self,
                 target_col: str = "target",
                 cols_to_drop: List[str] = None,
                 classification_task: bool = True,
                 clf_distinct_fl: bool = True,
                 discrete_thr: float = 0.025,
                 min_distinct_values: int = 2,
                 null_perc: float = 0.95,
                 thr_few_many_nulls: float = 0.75,
                 ) -> None:
        """

        Parameters
        ----------
        target_col: str
           name of the target/label column of the training dataframe
        cols_to_drop: List[str]
           list of columns to drop before feature selection computation (e.g. IDs like ndg or cntp_id, reference dates like snapshot_date or reference_date_m and additional cols like in_sample, timestamp, etc)
        classification_task: bool, optional
           flag to set the objective of the use case, (T -> Classification, F -> Other cases).
           Default: True
        clf_distinct_fl: bool, optional
           flag to keep for each features > 1 distinct value for each class. ALERT: Applicable only with 'classification_task' = True otherwise it is ignored.
           Default: True
        discrete_thr: float
           threshold for discrete features
        min_distinct_values: int
           minimum number of distinct values
        null_perc: float
           percentage to define a feature without null
        thr_few_many_nulls: float
           threshold to split in feature with few or many nulls
        """

        super(UniFeatureClassification, self).__init__(target_col=target_col, cols_to_drop=cols_to_drop)
        self.name = UniFeatureClassification.name
        self.classification_task = classification_task
        self.discrete_thr = discrete_thr
        self.min_distinct_values = min_distinct_values
        self.null_perc = null_perc
        self.thr_few_many_nulls = thr_few_many_nulls
        self.clf_distinct_fl = clf_distinct_fl

    def compute(
        self,  spark_session: pyspark.sql.SparkSession,
        df_train: Union[pyspark.sql.DataFrame, pd.DataFrame],
        **kwargs
    ) -> pyspark.sql.DataFrame:
        """Performing a function to the detect the features with many or few nulls eligible to be ignored

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
            Returns a pyspark dataframe object with features divided into various sets: continuous - few nulls, continuous - many nulls, discrete - few nulls, discrete - many nulls

        Raises
        ------
        NotImplementedError
            Whenever the input training dataset is neither a PySpark DataFrame, nor a Pandas DataFrame
        """

        # load params
        target_col = self.get_params()["target_col"]
        discrete_thr = self.get_params()["discrete_thr"]
        null_perc = self.get_params()["null_perc"]
        thr_few_many_nulls = self.get_params()["thr_few_many_nulls"]
        min_distinct_values = self.get_params()["min_distinct_values"]
        classification_task = self.get_params()["classification_task"]
        clf_distinct_fl = self.get_params()["clf_distinct_fl"]
        cols_to_drop = self.get_params()["cols_to_drop"]

        # Convert to PySpark if Pandas DataFrame and drop not needed cols
        # Raise an error if the input is not a 'pyspark.sql.DataFrame' or a 'pd.DataFrame'
        df_train = _convert_pandas_to_pyspark_df(spark_session, df_train, cols_to_drop)

        feat_cols = df_train.drop(target_col).columns

        df_train = (
            df_train
            .select(target_col, *(F.col(c).cast("double").alias(c) for c in feat_cols))  # cast all features to double
            # replacing NaN values with Nulls to have only one type of missing value
            .replace(float('nan'), None)
        )

        # convert data: N features x M row --> 2 features x (M x N) row
        df_stacked = (
            df_train
            .withColumn("flat", F.explode(F.array([F.struct(F.lit(c).alias("feature_name"), F.col(c).alias("feature_value")) for c in feat_cols])) )
            .select(target_col, "flat.feature_name", "flat.feature_value")
        )
        train_size = df_train.count()  # data row size

        # returns: spark df with the statistics of each feature
        # get information from data

        df_discrete_vs_continuous = (
            df_stacked
            .select('feature_name', 'feature_value')
            .groupBy('feature_name')
            .agg(
                F.countDistinct('feature_value').alias('count_distinct'),  # count of distinct values in distribution
                F.sum(F.when(F.col("feature_value").like('%.0') | F.col("feature_value").isNull(), 0).otherwise(1)).alias('only_integer'),  # Adds 1 for each value that has a decimal
                F.stddev('feature_value').alias('value_std'),  # computes the std for that feature
                F.sum(F.when(F.col("feature_value").isNull(), 1).otherwise(0)).alias('null_value_count')  # count how many nulls's there are in distribution
                )
            .withColumn("initial_size", F.lit(train_size))
            .withColumn("value_count", F.col('initial_size') - F.col('null_value_count'))  # numbers != None
            .withColumn("percentage_null", F.col('null_value_count') / F.col('initial_size'))  # Percentage between number of nulls and total
            .withColumn('perc_distinct', F.col('count_distinct') / F.col('initial_size'))  # percentage between distinct and total
            .withColumn('distribution', F.when((F.col('perc_distinct') < discrete_thr) &
                                               (F.col('only_integer') == 0), 'discrete').otherwise('continuous'))  # discrete or continuous
            .filter(F.col('percentage_null') < null_perc)
            .withColumn('null', F.when(F.col('value_count') >= (F.col('initial_size') * thr_few_many_nulls), 'few').otherwise('many'))
            )

        if ((classification_task) & (clf_distinct_fl)):
            # for each features > 1 distinct value for each class
            df_class_count_distinct_value = (
                df_stacked
                .dropna()  # In general in this univariate feature selection the nulls are considered
                .select(target_col, 'feature_name', 'feature_value')
                .groupBy('feature_name', target_col)
                .agg(F.countDistinct('feature_value').alias('count_distinct'))
                .withColumn("num_feature_name", F.lit(1))
                .groupBy('feature_name')  # group on each features
                .agg(
                    F.sum('count_distinct').alias('count_distinct'),  # add up the distinct values for each class
                    F.sum('num_feature_name').alias('num_class_exposed'),  # 2 if there are both classes in the count
                    )
                .filter(((F.col('num_class_exposed') == 2) & (F.col('count_distinct') <= 2)) | (F.col('num_class_exposed') == 1))
                .select('feature_name')
            )

            return (
                df_discrete_vs_continuous
                .join(df_class_count_distinct_value, ['feature_name'], 'left_anti')
                .drop('initial_size')
            )
        # returns: spark df with features divided into various sets: continuous - few nulls, continuous - many nulls, discrete - few nulls, discrete - many nulls
        # Minimum number of distinct values (default at least 2 distinct values)

        return (
            df_discrete_vs_continuous
            .filter(F.col('count_distinct') >= min_distinct_values)  # 2 distinct values within features at least
            .drop('initial_size')
            )