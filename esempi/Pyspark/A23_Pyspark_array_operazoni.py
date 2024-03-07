import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, DoubleType, StructField, StructType
from pyspark.sql.functions import udf
import os
from pyspark.sql.types import DecimalType
from decimal import Decimal
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

schema = StructType([
         StructField('column1', IntegerType()),
         StructField('column2', DecimalType(6, 2 ))])


dati = [
    (1, [Decimal(0.0), Decimal(0.0), Decimal(35.0), Decimal(48.0)]),
    (2, [Decimal(5.0), Decimal(15.0), Decimal(25.3), Decimal(35.9)])
]

schema = StructType([StructField('key', IntegerType(), nullable=False),
                     StructField('array_col', ArrayType(DecimalType(10,2)), nullable=False)])

df = spark.createDataFrame(dati , schema=schema)

df.show()


@udf(ArrayType(DoubleType()))
def delta_mol(array):
    array = list(map(float, array))
    differenze = [(array[i+1] / array[i] - 1)*100 if array[i] != 0.0 else 0.0 for i in range(len(array)-1) ]
    return differenze


dff = df.withColumn('delta_arry',delta_mol(F.col('array_col')))
dff.show()

# Altro modo    
@udf(ArrayType(DoubleType()))
def delta_mol(array):
    array = list(map(float, array))
    result = []
    for i in range(len(array) - 1):
        # Verifica che l'elemento successivo non sia zero per evitare divisioni per zero
        current = array[i]
        next_ = array[i+1]
        if  current != 0.0:
            result.append(((next_ - current) / current)*100)
        else:
            # Se l'elemento successivo è zero, il risultato della divisione sarà 0
            result.append(0.0)
    return result

dff = dff.withColumn('delta_arry_2',delta_mol(F.col('array_col')))


# al posto di esplode

# adesso mi creo tante colonne quanto sono gli elementi della colona array
max_length = dff.selectExpr("max(size(delta_arry_2))").collect()[0][0]

dff = dff.select("*",*[F.col("delta_arry_2")[i].alias(f"element_{i+1}") for i in range(max_length)])
dff.show()
