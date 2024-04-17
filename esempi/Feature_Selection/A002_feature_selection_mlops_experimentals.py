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
    .appName("PySpark Feature Selection")\
    .getOrCreate()
    
    
df = pd.read_csv('C:\\Travaux_2012\\compact.csv', sep=',')
ll = spark.createDataFrame(df)
ll.show()

spark.stop()