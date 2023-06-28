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
columns = ["name", "age_sum", "gender_sum"]


df = spark.createDataFrame(data, columns)

# voglio rinominare _sum

def col_replace(df):
    replacements = {c:c.replace('_sum','') for c in df.columns if '_sum' in c}
    df = df.select([col(c).alias(replacements.get(c, c)) for c in df.columns])
    return df
df = col_replace(df)
df.show()
    

spark.stop()


