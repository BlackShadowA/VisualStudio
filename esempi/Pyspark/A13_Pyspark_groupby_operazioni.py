import findspark
findspark.init()
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'ndg':[10,10,20,30,30],
    'operazioni':[1,2,3,4,5],
    'ammontari':[50,20,10,80,9]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))


funs = [sum, F.count]
cols = ["operazioni", "ammontari"]
aggregazione = ll.groupby('ndg')\
    .agg(*[f(c).alias(f'{c}_{f.__name__}') for c in cols for f in funs])

aggregazione.show()

# Metodo simile

