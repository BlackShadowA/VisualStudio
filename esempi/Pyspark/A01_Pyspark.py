import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

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

# isin(valori e null funziona solo cosi
ll.filter(F.col('valori').isin(10) | F.col('valori').isin('NaN')).show()

import pyspark
def funzione(df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:

    return df, df

ll, pp = funzione(ll)
print("pp")
pp.show()
print("ll")
ll.show()

data = [['Scott', 50], ['Jeff', 45], ['Thomas', 54],['Ann',34]] 
 
# Create the pandas DataFrame 
pandasDF = pd.DataFrame(data, columns = ['Name', 'Age'])

ll = spark.createDataFrame(pandasDF)
ll.show()


spark.stop()