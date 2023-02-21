
import findspark
findspark.init()
from pyspark.sql.session import SparkSession
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import datetime as dt
from pyspark.sql.types import StringType




spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()



from pyspark.sql.functions import *
df = spark.createDataFrame([[2017,9,3 ],[2015,5,16]],['year', 'month','date'])
df = df.withColumn('timestamp',to_date(concat_ws('-', df.year, df.month,df.date))).cache()
df.show()


import datetime
my_date =  datetime.date(2015, 8, 2)
current_date = datetime.datetime.today().date()
print(current_date)
print(my_date)

def condizione(data):
        
        if data >= datetime.date(2015, 8, 2):
                return 'ok'
        else:
                return 'ko'

condizioni = F.udf(condizione, StringType())

ll = df.withColumn('pp', condizioni(F.col('timestamp')))
ll.show()
ll.printSchema()

spark.stop()