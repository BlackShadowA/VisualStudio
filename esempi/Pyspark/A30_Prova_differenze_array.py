import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
from datetime import datetime
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import expr


spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

data = [("2021-01-01", ["2021-02-01", "2021-03-01"]),
        ("2022-03-15", ["2022-04-15", "2022-05-15"]),
        ("2023-05-20", ["2023-06-20", "2023-07-20"])]

df = spark.createDataFrame(data, ["data", "array_col"])
df.show()


def diff_in_months(start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    return delta

df = df.withColumn("diff_months", expr("diff_in_months(data, array_col[0])"))
df.show()