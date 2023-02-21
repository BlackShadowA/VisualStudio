import findspark
findspark.init()
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,IntegralType
from pyspark.sql import SparkSession
from pyspark.sql.session import SparkSession
import pyspark


spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
sc.setLogLevel("WARN")

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'customer_key': ['2280765901', '66001', '12942501', '2280766201', '2280766301', '2280766401', '2280767701'],
    'amount': [30, 90, 35, None, 20, 25, 35],
    'transaction': [20, 27, 35, 33, 18, 20, 35],
    'volume': [18, 90, 35, 150, 20, 250, 35],
}


ll = spark.createDataFrame(pd.DataFrame(my_dict)).cache() # va molto più veloce
ll.show()

# ############################################# #
# Massimo & Minimo                              #
# ############################################# #
# Primo Metodo
colonne = ['amount', 'volume']
newdf = ll.withColumn('array_columns', F.array(colonne))\
          .withColumn('max', F.sort_array("array_columns", False)[0])\
          .withColumn('min', F.sort_array("array_columns", True)[0])

newdf.show()
# Secondo Metodo
colonne = ['amount', 'volume']
newdf = ll\
    .withColumn('max', F.greatest(*colonne))\
    .withColumn('min', F.least(*colonne))

newdf.show()

# ############################################# #
# Somma                                         #
# ############################################# #
# Primo metodo
newdf = ll.withColumn('total_somma_colonne', sum(ll[col] for col in ll.columns if col !='customer_key'))
newdf.show()

# Secondo Metodo
colonne = [col for col in ll.columns if col !='customer_key']
df = ll.withColumn('array_columns', F.array(colonne))\
    .select('*', sum([F.col('array_columns').getItem(i) for i in range(len(colonne))]).alias('Total'))
df.show()



spark.stop()





