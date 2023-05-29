import findspark
findspark.init()
from pyspark.sql import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

# Crea la sessione Spark
spark = SparkSession.builder.getOrCreate()

data = [('Scott', 50), ('Jeff', 45), ('Thomas', 54),('Ann',34)] 
df=spark.createDataFrame(data,["name","age"]) 
df.show()

# creo una funzione che mi restituisce il numero di colonne e righe
def sparkShape(dataFrame):
    return (dataFrame.count(), len(dataFrame.columns))

import pyspark
pyspark.sql.dataframe.DataFrame.shape = sparkShape

print(df.shape())


def schiaccio(input_df):
    from pyspark.sql import functions as F
    # Seleziona solo le colonne numeriche
    numeric_cols = [
        col for col, dtype in input_df.dtypes if (dtype in ('int', 'bigint', 'float', 'double')) & (col not in ('id'))
        ]
    # Calcola le medie delle colonne numeriche
    averages_df = input_df.select([
         F.mean(col).alias(f'avg_{col}') for col in numeric_cols
    ])

    return averages_df

import pyspark
pyspark.sql.dataframe.DataFrame.grupppoVarNumeriche = schiaccio

ll = df.grupppoVarNumeriche()
ll.show()
