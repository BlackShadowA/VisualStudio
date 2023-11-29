import findspark
findspark.init()
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import sum, col
from pyspark.sql.session import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'




spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'nomi':['aa','bb','cc','dd','ee'],
    'ndg':[1,2,3,4,5],
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

# primo metodo

def extract_column_values_to_list(df, column_name):
    values_list = df.select(column_name).rdd.flatMap(lambda x: x).collect()
    return values_list


features = extract_column_values_to_list(ll, 'nomi')
print(features)

# secondo metodo

colonne = [row['nomi'] for row in ll.select(F.col("nomi")).collect()]
print(colonne)