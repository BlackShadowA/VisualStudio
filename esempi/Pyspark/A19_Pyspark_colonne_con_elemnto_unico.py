import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
from pyspark.sql.functions import col

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()
    

def columns_with_single_value(df):
    # Calcola il numero di valori unici per ciascuna colonna
    unique_counts = df.agg(*[(F.countDistinct(c).alias(c + '_count')) for c in df.columns])

    # Filtra le colonne con conteggio unico pari a 1
    single_value_columns = [c for c in df.columns if unique_counts.select(col(c + '_count')).first()[0] == 1]

    return single_value_columns


# Esempio di DataFrame di input
data = [(1, 3, 1), (2, 3, 2), (3, 3, 3)]
columns = ["col1", "col2", "col3"]
df = spark.createDataFrame(data, columns)
df.show()

# Chiama la funzione per ottenere le colonne con un solo valore
single_value_cols = columns_with_single_value(df)

print("Colonne con un solo valore:", single_value_cols)