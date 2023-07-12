import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,IntegralType
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
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


spark = SparkSession.builder.master("local[*]").appName("Mio Test").getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'ProductType':['CreditExpress','CreditExpress','CreditExpress - Compact','CreditExpress - Compact'],
    'amount':[40000,30000,40000,100000],
    'tasso': [7.25,7.25,8.0,6.0]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))
ll.show()

# Primo metodo
from itertools import chain

vol_o_w_one = { 40001: 'MOL_DOUBLE_COUNTING',
                30000: 'MOL_IMPIEGHI_CLIENTELA',
                100000: 'MOL_RACCOLTA'
            }

mapping_expr = F.create_map([F.lit(x) for x in chain(*vol_o_w_one.items())])

# Secondo metodo con Replacle
df = ll.withColumn("value", mapping_expr.getItem(F.col("amount")))
df.show()