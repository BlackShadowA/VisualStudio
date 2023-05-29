import findspark
findspark.init()
from pyspark.sql import SparkSession
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

# Crea la sessione Spark
spark = SparkSession.builder.getOrCreate()


# Crea un DataFrame di esempio
data = [
    (1, 10.0, 100),
    (2, 20.0, 200),
    (3, 30.0, 300),
    (4, 40.0, 400)
]
df = spark.createDataFrame(data, ['id', 'col1', 'col2'])

df.show()



from pyspark.sql import functions as F

class SparkDataFrameAverages:
    def __init__(self):
        self

    def calculate_averages(self, input_df):
        # Seleziona solo le colonne numeriche
        numeric_cols = [
            col for col, dtype in input_df.dtypes if (dtype in ('int', 'bigint', 'float', 'double')) &  (col not in ('id'))
        ]
        
        # Calcola le medie delle colonne numeriche
        averages_df = input_df.select([
            F.mean(col).alias(f'avg_{col}') for col in numeric_cols
        ])

        return averages_df

# Crea un'istanza della classe SparkDataFrameAverages
averages_calculator = SparkDataFrameAverages()

# Calcola le medie delle variabili numeriche
averages = averages_calculator.calculate_averages(df)

# Visualizza il DataFrame con le medie
averages.show()