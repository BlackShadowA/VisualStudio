import findspark
findspark.init()
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import sum, col
from pyspark.sql.session import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'




spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'stock_afi':[10,10,20,30,30],
    'ndg':[1,2,3,4,5],
    'stock_afi_indiretta':[20,30,40,50,60],
    'stock_afi_indiretta_mol':[50,20,10,80,9]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

selected_columns = [column for column in ll.columns if column.startswith("stock_")]

selected_var = selected_columns
funs = [F.sum]
exprs = [f(col(c)).alias(f'{c}_{f.__name__}') for f in funs for c in selected_var]

afi = (
     ll
     .groupBy('ndg')
     .agg(*exprs)
    )

afi.show()