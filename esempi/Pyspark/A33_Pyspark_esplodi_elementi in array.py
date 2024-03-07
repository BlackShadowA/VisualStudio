import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, DoubleType
from pyspark.sql.functions import udf
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")



dati = [
    (1, [0, 0, 35, 48, 85, 68, 70, 80, 93, 100, 110, 120]),
    (2, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]),
    (3, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
]

df = spark.createDataFrame(dati, ["chiave", "array_col"])
df.show(truncate=False)

# adesso mi creo tante colonne quanto sono gli elementi della colona array
max_length = df.selectExpr("max(size(array_col))").collect()[0][0]

df = df.select("*",*[F.col("array_col")[i].alias(f"element_{i+1}") for i in range(max_length)])
df.show()