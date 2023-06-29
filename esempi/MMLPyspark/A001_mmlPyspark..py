import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.10.2")\
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")\
    .getOrCreate()
    
import synapse.ml

print(f"Versione Pyspark = {spark.version}")

my_dict = dict(key=[10, 20, 30], valori=[50, None, 10])

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()



