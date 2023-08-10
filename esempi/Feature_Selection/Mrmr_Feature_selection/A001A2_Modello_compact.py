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
    
# modello compact
# filename_compact = 'C:\\Users\\ur00601\\Downloads\\Feature importance mrmr.csv'
filename_compact = 'C:\\Users\\ur00601\\Downloads\\trian_test.csv'
df3 = spark.read.csv(filename_compact , header=True,inferSchema=True, sep=',')\
           .drop('n_pp', 'customer_key', 'dt_riferimento')

from woe_pyspark import var_type,WOE
selected_features = mrmr_classif(df = df3, target_column="label", K=10)
print(selected_features)

# Woe e IV delle variabili selezionate
'''
target = 'label'
max_bin = 5
ll ,pp = WOE(df3, selected_features, target, max_bin)
ll = ll.sort('varname', 'start')
ll.show()
pp.show()
plotBinsSummary(ll.toPandas(), var_name = 'importo_erogato_last_pp')
'''