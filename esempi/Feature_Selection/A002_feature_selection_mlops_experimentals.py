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


from unicredit_mlops_experiments.experiment import ExperimentUtils
from typing import Dict, Any
import lightgbm as lgb

EXPERIMENT_NAME = 'baseline'
BASE_PATH = "/Users/UR00601/CAR Nuovo/feature_selection_output"
TARGET_VARIABLE = 'target'
BASE_INPUT = "/Users/UR00601/CAR Nuovo/workbook-output/"

FEATURES_FAMILIES = [
        'cntp_',  # aggiungere apice a nuovo run master
        'afi__',
        'card__',
        'bancass__',
        'account__',
        'margin__',
        'mifid__',
        'multibank__',
        'mutui__',
        'loan__',
        'transaction__'
]

    # features that will be add on top at the feature selection phase
MUST_HAVE_FEATS = [  ]

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

FEATURE_CLASSIFICATION_METHOD = ExperimentUtils(method_name="nulls_detector").get_feature_selection()

config_experiment: Dict[str, Any] = {
    'standard': {
        "name": EXPERIMENT_NAME,
        "train": f"{BASE_INPUT}",
        "experiments_location": f"{BASE_PATH}/experiments",
        "output_location": f"{BASE_PATH}/model",
        'feat_selection': [
            FEATURE_CLASSIFICATION_PARAMS
        ],
        "must_have_feats": MUST_HAVE_FEATS
        },
}

config_experiment: Dict[str, Any] = {
    'standard': {
        "name": EXPERIMENT_NAME,
        'feat_selection': [
            FEATURE_CLASSIFICATION_PARAMS
        ],
        
        },
}

config_dict = config_experiment
for _, experiment in config_dict.items():
    pp_feats = [s for s in snap.columns if ('loan__active' in s or 'loan_personali' in s)]
    train_all_feats = snap.drop(*pp_feats)
    
    feature_classification = FEATURE_CLASSIFICATION_METHOD
    feature_classification.set_params(**config['feat_selection'][0])

    feats_classifed = feature_classification.compute(
            spark,
            train_all_feats
        )
    feats_classifed.show()

spark.stop()