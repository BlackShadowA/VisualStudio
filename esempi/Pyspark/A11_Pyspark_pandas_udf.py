import findspark
findspark.init()
import pandas as pd
from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as F
#quando fai girare in Terminale
import os

import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()
# ottimizza le pandas udf e le udf e lo spotamento da Pyspark a Python
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')


my_dict = {
    'key':[10,20,30,10 ,30 ],
    'valori':[50,None,10, 50, 60]
}

df = spark.createDataFrame(pd.DataFrame(my_dict)) 
df.show()  
'''
df = spark.createDataFrame(
    [[1, "a string", ("a nested string",)]],
    "long_col long, string_col string, struct_col struct<col1:string>")

df.show()
df.printSchema()
'''
# ############################################### #
# Seriest to Series
# ############################################### #
from pyspark.sql.functions import PandasUDFType

# Style spark 2.
@pandas_udf("long", PandasUDFType.SCALAR)
def pandas_plus_one(v):
    return v + 1

df_old = df.withColumn('somma', pandas_plus_one(F.col('valori')))
df_old.show()

# New Style Spark 3.
@pandas_udf('long')
def pandas_plus_one(s: pd.Series) -> pd.Series:
    return s + 1

df_new = df.withColumn('somma', pandas_plus_one(F.col('valori')))
df_new.show()

# ############################################### #
# Scalar Iter
# ############################################### #
@pandas_udf("long", PandasUDFType.SCALAR_ITER)
def pandas_plus_one(vv):
    return map(lambda v: v + 1, vv)

df_new = df.withColumn('somma', pandas_plus_one(F.col('valori')))
df_new.show()

# o anche
@pandas_udf('long', PandasUDFType.SCALAR_ITER)
def pandas_plus_one(iterator):
    return map(lambda s: s + 1, iterator)

df_new = df.withColumn('somma', pandas_plus_one(F.col('valori')))
df_new.show()


spark.stop()

