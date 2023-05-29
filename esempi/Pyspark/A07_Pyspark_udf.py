import findspark
findspark.init()
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StructField, FloatType

import pyspark
from pyspark.sql.session import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

'''
import os
os.environ["JAVA_HOME"]= 'C:\\Progra~1\\Java\\jre8'
os.environ["SPARK_HOME"]= 'C:\\Travaux_2012\\spark'
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\hadoop'
'''
#Spark Ui http://localhost:4040

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()
# ottimizza le pandas udf e le udf e lo spotamento da Pyspark a Python
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'key':[10,20,30],
    'valori':["AA_05_bb",'AA_5_bb','AA_05_bb']
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

def f(x, h):
    return x + 100 + h
convert = udf(lambda x , y: f(x,y), IntegerType())

ll = ll.withColumn('app', convert(F.col('key'),F.lit(100)))
ll.show()





