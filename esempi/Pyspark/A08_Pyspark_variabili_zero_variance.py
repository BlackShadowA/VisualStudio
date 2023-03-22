import findspark
findspark.init()
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StructField, FloatType
from pyspark.sql.session import SparkSession
import pyspark


spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

# ottimizza le pandas udf e le udf e lo spotamento da Pyspark a Python
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

print(f"Versione Pyspark = {spark.version}")

# Carico dataset
df = pd.read_excel("C:\\Travaux_2012\\Esempi_python\\dataset\\ccdata.xls")
df = spark.createDataFrame(df)
df.show()

# Seleziono le variabili numeriche
variabili_numeriche =[nome for nome, tipo in df.dtypes if tipo !='string'] 
print(variabili_numeriche)


    
diz = {}
for nf in variabili_numeriche:
    diz[nf] = 'variance'
print(diz)

# calcolo la variance per tutte le variabili numeriche
varian = df.agg(diz)
varian.show()

# lo porto in Pandas

varian_pandas = varian.toPandas()

names = []
for c in varian_pandas.columns:
    if varian_pandas[c].values == 0:
        names.append(c.replace('variance(','').replace(')',''))  

print(names)
