import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
from Mrmr_pyspask import mrmr_classif, mrmr_regression
from woe_pyspark import var_type,WOE
from plot_woe import plotBinsSummary
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'



#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

filename = 'C:\\Users\\ur00601\\Downloads\\Monotonic-Optimal-Binning-main\\data\\german_data_credit_cat.csv'
df = spark.read.csv(filename, header=True,inferSchema=True, sep=';')\
    .withColumn('default', F.col('default') + F.lit(-1))
    
# classificazione    
selected_features = mrmr_classif(df = df, target_column="default", K=3)
print(selected_features)
target = 'default'
max_bin = 5
ll ,pp = WOE(df, selected_features, target, max_bin)
ll.show()
plotBinsSummary(ll.toPandas(), var_name = 'Durationinmonth')
'''
#  Regressione
file = "C:\\Travaux_2012\\Esempi_python\\train.csv"
df2 = spark.read.csv(file, header=True,inferSchema=True, sep=',')

selected_features_regression = mrmr_regression(df = df2, target_column="SalePrice", K=3)
print(selected_features_regression)



# modello compact
filename_compact = 'C:\\Users\\ur00601\\Downloads\\Feature importance mrmr.csv'
df3 = spark.read.csv(filename_compact , header=True,inferSchema=True, sep=',')\
           .drop('n_pp')

selected_features = mrmr_classif(df = df3, target_column="label", K=3)
print(selected_features)

# Woe e IV delle variabili selezionate
target = 'label'
max_bin = 5
ll ,pp = WOE(df3, selected_features, target, max_bin)
ll = ll.sort('varname', 'start')
ll.show()
pp.show()
'''