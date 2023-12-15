import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'key':[10,20,30],
    'valori':[50,None,10],
    'ss': [50,None,10]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

ss = [f"{count+1}_{item}" for count,item in enumerate(ll.columns)]
print(ss)

ll = ll.toDF('ss', *[f'n_{c}' for c in ll.columns if c not in ['ss']])
ll.show()