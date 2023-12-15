import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
from pyspark.sql.functions import col
import itertools


#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()
    
    
# documentazione qui hai diversi esempi
# https://stackoverflow.com/questions/51082758/how-to-explode-multiple-columns-of-a-dataframe-in-pyspark

data = [("Alice", 25, "F", 30), ("Bob", 30, "M", 90), ("Charlie", None, "M", 100)]
columns = ["name", "age", "gender", "ammontare"]


df = spark.createDataFrame(data, columns)
df.show()

df = df.select('name', *[col(c).alias(f"{count+1}_{c}") for count, c in enumerate(df.columns) if c not in ['name']])

df.show()