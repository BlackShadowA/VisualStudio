import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
from pyspark.sql.functions import col

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()
    
    
# documentazione qui hai diversi esempi
# https://stackoverflow.com/questions/51082758/how-to-explode-multiple-columns-of-a-dataframe-in-pyspark

data = [("Alice", 25, "F"), ("Bob", 30, "M"), ("Charlie", None, "M")]
columns = ["name", "age", "gender"]


df = spark.createDataFrame(data, columns)
df.show()

from pyspark.sql.functions import array, explode, struct
# from functools import reduce



def stack_columns(df, key_col):
    columns = [c for c in df.columns if c != key_col]
    df = df.withColumn("nomi_colonna", F.array(*map(F.lit, columns)))\
            .withColumn("valori_colonna", F.array([col(c) for c in columns]))\
            .withColumn("creo_array_multiplo", F.arrays_zip("nomi_colonna", "valori_colonna"))\
            .withColumn("new_", F.explode("creo_array_multiplo"))\
            .select(col(key_col), F.col("new_.nomi_colonna").alias("nomi_colonna"), 
                                  F.col("new_.valori_colonna").alias("valori_colonna"))

    return df

stacked_df = stack_columns(df, "name")
stacked_df.show()

  

spark.stop()





