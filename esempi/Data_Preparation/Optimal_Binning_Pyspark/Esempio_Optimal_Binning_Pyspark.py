import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
from woe_pyspark import var_type, execute_woe
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'



#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

filename = 'C:\\Users\\ur00601\\Downloads\\Monotonic-Optimal-Binning-main\\data\\german_data_credit_cat.csv'
df = spark.read.csv(filename, header=True,inferSchema=True, sep=';')

char_vars, num_vars = var_type(df)
print(char_vars)
print(num_vars)

target_variable_name = 'default'
final_vars  = char_vars + num_vars
output, iv = execute_woe(df, target_variable_name, final_vars)

print(output)
print(iv)