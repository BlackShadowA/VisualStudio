import findspark
findspark.init()
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

spark = SparkSession\
    .builder\
    .appName("PySpark Feature Selection")\
    .getOrCreate()
    
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)
  
#df = pd.read_csv('C:\\Travaux_2012\\compact.csv', sep=',')
df = pd.read_excel("C:\\Travaux_2012\\compact.xlsx")
print(df)


snap = spark.createDataFrame(df)
snap.show(truncate=False)

#from feature_selection_mio.fs_abstract import UniFeatureClassification
from feature_selection_mio.univariate.UniFeatureClassification import UniFeatureClassification
from typing import Dict, Any
TARGET_VARIABLE = 'target'
#COLS_TO_DROP = ["cntp__cust_age"]

FEATURE_CLASSIFICATION_PARAMS: Dict[str, Any] = {
    "classification_task": True,
    "clf_distinct_fl": True,
    "cols_to_drop":False,
    "discrete_thr": 0.025,
    "min_distinct_values": 2,
    "null_perc": 0.95,
    "std_thr": 0.001,
    "thr_few_many_nulls": 0.75,
    "target_col": TARGET_VARIABLE
    }

feat_imp = UniFeatureClassification().set_params(**FEATURE_CLASSIFICATION_PARAMS)
feats_classifed = feat_imp.compute(spark, snap)
feats_classifed.show()


    
    
spark.stop()


