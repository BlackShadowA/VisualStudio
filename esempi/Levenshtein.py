import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'key':[10,20,30],
    'nome_1':['GAETANO MAURO','ROBERTA','ROBERTA BALBO'],
    'nome_2':['GAETANO MAURO','GRAMONDO','BALBO ROBERTA']
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))\
          .withColumn('dimensione', F.size(F.split(F.col("nome_2"), " ")))\
          .withColumn('inverso', F.reverse(F.col("nome_2")))
ll.show()



# indice di similarità di levenshtein

def similarity(df, col_name_1, col_name_2):

    df = df.withColumn("levenshtein_dist", F.levenshtein(df[col_name_1], df[col_name_2]))

    df = df.withColumn("l1", F.length(df[col_name_1])).withColumn("l2", F.length(df[col_name_2]))
    df = df.withColumn("max", F.when(df["l1"] > df["l2"], df["l1"]).otherwise(df["l2"]))

    df = df.withColumn("similarity", F.lit(1)-(df["levenshtein_dist"]/df["max"]))
    df = df.withColumn("similarity", F.when((df["l1"] == 0) | (df["l2"] == 0), F.lit(0)).otherwise(df["similarity"]))
    df = df.withColumn("levenshtein_dist", F.when((df["l1"] == 0) | (df["l2"] == 0), F.lit(1000)).otherwise(df["levenshtein_dist"]))

    df = df.drop("l1", "l2", "max")

    return df

ver = similarity(ll, 'nome_1', 'nome_2')
ver.show()