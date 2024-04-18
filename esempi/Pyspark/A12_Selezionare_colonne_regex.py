import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
import pyspark.sql.functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")


my_dict = {
    'ProductType':['CreditExpress','CreditExpress','CreditExpress - Compact','CreditExpress - Compact'],
    'ff_A':[4000,30000,40000,100000],
    'ff_N': [7.25,7.25,8.0,6.0],
    'll_A':[5000,20000,30000,100000],
    'll_N': [7.25,7.25,8.0,6.0]

}

df = spark.createDataFrame(pd.DataFrame(my_dict))
df.show()

import re


# cerco prima le colonne che contengono _A
amt = [c for c in df.columns if re.match(r'\w*_A$', c)]
print(amt)
# puoi anche usare
amt = [c for c in df.columns if "_A" in c]
#sostituisco _A con blanck
amt_rid = [re.sub('_A$','',s) for s in amt]
print(amt_rid)

# cerco prima le colonne che contengono _N
num = [c for c in df.columns if re.match(r'\w*_N$', c)]
print(num)
#sostituisco _N con blanck
num_rid = [re.sub('_N$','',s) for s in num]
print(num_rid)

# Adesso mi calcolo le medie

for_avg = list(set(num_rid).intersection(amt_rid))
print( '########')
print( for_avg)

for c in for_avg:
    df = df.withColumn(c+'_AVG', F.col(c+'_A')/F.col(c+'_N'))
 
df.show()



spark.stop()


