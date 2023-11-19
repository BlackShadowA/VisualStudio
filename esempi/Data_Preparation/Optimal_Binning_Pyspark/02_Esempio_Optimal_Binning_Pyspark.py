import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
from woe_pyspark import var_type,WOE
from plot_woe import plotBinsSummary
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

filename = 'C:\\Users\\ur00601\\Downloads\\Monotonic-Optimal-Binning-main\\data\\german_data_credit_cat.csv'
df = spark.read.csv(filename, header=True,inferSchema=True, sep=';')\
    .withColumn('default', F.col('default') + F.lit(-1))

asp = df.groupby('default').agg(F.count('*').alias('n')).show()
char_vars, num_vars = var_type(df)
print(char_vars)
print(num_vars)

target = 'default'
max_bin = 5
ll ,pp = WOE(df, num_vars, target, max_bin)
ll = ll.sort('varname', 'start')
ll.show()
pp.show()


spark.stop()