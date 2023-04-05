import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'key':[10,20,30],
    'valori':[50,None,10]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()