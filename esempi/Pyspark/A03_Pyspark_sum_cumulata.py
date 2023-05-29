import findspark
findspark.init()
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

#import cloudpickle as pickle
'''
import os
os.environ["JAVA_HOME"]= 'C:\\Progra~1\\Java\\jre8'
os.environ["SPARK_HOME"]= 'C:\\Travaux_2012\\spark'
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\hadoop'
'''


spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'struct':[10,10,20,30,30],
    'ndg':[1,2,3,4,5],
    'PP':[20,30,40,50,60],
    'fab':[50,20,10,80,9]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

from pyspark.sql import Window

pp = [F.desc('ndg'),F.asc('PP')]
windowval = (Window.partitionBy('struct').orderBy(pp)
             .rangeBetween(Window.unboundedPreceding, 0))
df_w_cumsum = ll.withColumn('cum_sum', F.sum('fab').over(windowval))
df_w_cumsum.show()

spark.stop()