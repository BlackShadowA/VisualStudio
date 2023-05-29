import findspark
findspark.init()
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'key':[10,20,30],
    'valori':["AA_05_bb",'AA_5_bb','AA_05_bb']
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

ff = ll.withColumn('pippo',F.instr(F.col("valori"), '_'))\
    .withColumn('aa', F.substring_index(F.col("valori"), '_', 2))\
    .withColumn('bb', F.substring_index(F.col("aa"), '_', -1))
ff.show()