# ##################################### #
# Nuovo XGboost Paralellizzato nativo   #
# ##################################### #
import findspark
findspark.init()
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")


import pandas as pd
df = pd.read_excel("C:\\Travaux_2012\\Esempi_python\\ccdata.xls")

#Per adesso elimino le colonne categoriche
column_string = [ col  for col, dt in df.dtypes.items() if dt == object] + ['ID']
print(column_string)
df = df.drop(column_string, axis = 1)

ll = spark.createDataFrame(df)

# Trasformo le colonne in Double
def schema(df):
    schema = {}
    for col in df.schema:
        schema[col.name] = col.dataType
    cols2 = []
    for c, t in schema.items():
        if str(t).startswith('Decimal') | str(t).startswith('Long') |  str(t).startswith('Int'):
            cols2.append(F.col(c).cast('Double'))
        else:
            cols2.append(F.col(c))
    return df.select(cols2)

df = schema(ll).na.fill(0)
df.printSchema()


label = "default payment next month"
feature_names = [x.name for x in df.schema if x.name != label]


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler().setInputCols(feature_names).setOutputCol("features")
df_ = vectorAssembler.transform(df)

import pyspark
from xgboost.spark import SparkXGBClassifier

modello = SparkXGBClassifier(
    features_col="features",
    label_col=label,
    num_workers=1 # in locale metti 1
)
model = modello.fit(df_)

# predict on test data
predict_df = model.transform(df_)
predict_df.show(truncate= False)




spark.stop()