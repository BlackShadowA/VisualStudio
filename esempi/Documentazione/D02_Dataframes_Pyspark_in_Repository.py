'''
Per creare un datafraes pyspark in repository:

Importante: 

•	Devi mettere lo schema altrimenti va in errore

'''

from pyspark.sql import functions as F
from transforms.api import transform,  Input, Output
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

@transform(
    processed=Output("/Users/UR00601/analisi/datasets/Incremental"),
)
def compute(ctx, processed):

    data = [("James", "", "Smith", "36636", "M", 3000),
            ("Michael", "Rose", "", "40288", "M", 4000),
            ("Robert", "", "Williams", "42114", "M", 4000),
            ("Maria", "Anne", "Jones", "39192", "F", 4000),
            ("Jen", "Mary", "Brown", "", "F", -1)
    ]

    schema = StructType([ 
        StructField("firstname", StringType(), True), 
        StructField("middlename", StringType(), True), 
        StructField("lastname", StringType(), True), 
        StructField("id", StringType(), True), 
        StructField("gender", StringType(), True), 
        StructField("salary", IntegerType(), True) 
    ])

    df = ctx.spark_session.createDataFrame(data=data, schema=schema)
    return processed.write_dataframe(df)
