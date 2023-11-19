import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
import os
from woe_pyspark import var_type,WOE
from plot_woe import plotBinsSummary
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")

filename = 'C:\\Users\\ur00601\\Downloads\\Monotonic-Optimal-Binning-main\\data\\german_data_credit_cat.csv'
df = spark.read.csv(filename, header=True,inferSchema=True, sep=';')\
    .withColumn('default', F.col('default') + F.lit(-1))

asp = df.groupby('default').agg(F.count('*').alias('n')).show()
char_vars, num_vars = var_type(df)
print(char_vars)
print(num_vars)

target = 'default'
max_bin = 5
ll ,pp = WOE(df, num_vars, target, max_bin)
ll = ll.sort('varname', 'start')
ll.show()
pp.show()

pandas_df = ll.toPandas()
# prendo le colonne con IV più alto di un valore e vedo se sono correlate tra loro
ivv = pp.toPandas()
ss = ivv.loc[ivv['IV'] > 0.010]
top_cols = ss["varname"].values.tolist()
print(top_cols)

# Correlazione delle migliori Iv
df_pandas = df.toPandas()
corr_df = df_pandas[top_cols].corr()
plt.figure(figsize=(25, 9))
sns.heatmap(corr_df,annot=True ,cmap=sns.color_palette("BrBG",2));
plt.show()


print('Bins Size Base')
plotBinsSummary(pandas_df, var_name = 'Durationinmonth')


# seleziono tra le correlate quelle con IV migliore

# da fare https://www.kaggle.com/code/gopidurgaprasad/amex-credit-score-model
# converto in Series 

iv_score_dict = ss.to_dict('records')
print(iv_score_dict)


def drop_feature_selection(row, col, corr, row_iv, col_iv):
    if row_iv >= col_iv:
        return col
    else:
        return row
    

cor_matrix = df_pandas[top_cols].corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
corr_df = upper_tri.stack().reset_index()
corr_df.columns = ['row', 'col', 'corr']
corr_df = corr_df.drop_duplicates()
corr_df = corr_df.sort_values('corr', ascending=False)
corr_df = corr_df.query("corr >= 0.8")
corr_df['row_iv'] = corr_df['row'].map(iv_score_dict)
corr_df['col_iv'] = corr_df['col'].map(iv_score_dict)

corr_df['drop_feature'] = corr_df.apply(lambda x: drop_feature_selection(x['row'], x['col'], x['corr'], x['row_iv'], x['col_iv']), axis=1)
print(corr_df)
