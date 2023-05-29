import findspark
findspark.init()
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
#import cloudpickle as pickle

#Spark Ui http://localhost:4040
from pyspark.sql.session import SparkSession
import pyspark
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'KEY':[10,20,30],
    'VALORI':[50,None,10]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

ll1 = ll.toDF(*[c.lower() for c in ll.columns])
ll1.show()



spark.stop()