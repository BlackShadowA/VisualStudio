import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StructField, FloatType, StringType, ArrayType, DoubleType



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

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()



# indice di similarità di levenshtein

def similarity(df, col_name_1, col_name_2):
    from pyspark.sql.functions import udf
    from pyspark.sql.types import  StringType

    def f(x):
        y = x.split(' ')
        r = y[::-1]
        z = ' '.join(r)
        return z
    # Potrebbe il nome ed il cognome essere invertiti
    inverto = udf(lambda x : f(x), StringType())
    # Inverto Nome e Cognome
    df = df.withColumn('inverto_ordine', inverto(df[col_name_2]))

    df = df.withColumn("levenshtein_one", F.levenshtein(df[col_name_1], df[col_name_2]))\
           .withColumn("levenshtein_two", F.levenshtein(df[col_name_1], F.col('inverto_ordine')))\
           .withColumn("levenshtein_dist", F.least("levenshtein_one", "levenshtein_two"))\
           .drop("levenshtein_one", "levenshtein_two")
    
    df = df.withColumn("l1", F.length(df[col_name_1])).withColumn("l2", F.length(df[col_name_2]))
    df = df.withColumn("max", F.when(df["l1"] > df["l2"], df["l1"]).otherwise(df["l2"]))

    df = df.withColumn("similarity", F.lit(1)-(df["levenshtein_dist"]/df["max"]))
    df = df.withColumn("similarity", F.when((df["l1"] == 0) | (df["l2"] == 0), F.lit(0)).otherwise(df["similarity"]))
    df = df.withColumn("levenshtein_dist", F.when((df["l1"] == 0) | (df["l2"] == 0), F.lit(1000)).otherwise(df["levenshtein_dist"]))

    df = df.drop("l1", "l2", "max", "inverto_ordine")
    return df

ver = similarity(ll, 'nome_1', 'nome_2')
ver.show()

# Metodo che lavora sulle liste ,secondo me questo è migliore
# Trasformo in Array

ll = ll.withColumn('nome_1' ,F.array(F.col('nome_1')))\
       .withColumn('nome_2' ,F.array(F.col('nome_2')))
ll.show()


def similarity_list(col_name_1, col_name_2):

    def suddividi_nomi(lista_completa):
        nomi_singoli = []

        for nome_completo in lista_completa:
            nomi_singoli.extend(nome_completo.split())

        return nomi_singoli

    lista_nomi_singoli1 = suddividi_nomi(col_name_1)
    lista_nomi_singoli2 = suddividi_nomi(col_name_2)
    result = all(elem in lista_nomi_singoli1 for elem in lista_nomi_singoli2)
    
    app = 0.0
    
    if result == True:
        app = 1.0

    return  app

levenshtein_udf = udf(similarity_list, DoubleType())
result_df = ll.withColumn("levenshtein_distance_nuova", levenshtein_udf(ll["nome_1"], ll["nome_2"]))
result_df.show()

