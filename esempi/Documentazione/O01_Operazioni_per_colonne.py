
'''
Operazioni per Colonne:

•	Massimo & Minimo
•	Somma  COLONNE
•	Moltiplicazione
•	Media

Ho il seguente DataFrames:
'''

from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F 
import pandas as pd 
from pyspark.sql.types import LongType, IntegerType, DoubleType, FloatType
from pyspark.sql.functions import udf

sc = SparkContext()
spark = SQLContext(sc)

my_dict = {
    'customer_key': ['2280765901', '66001', '12942501', '2280766201', '2280766301', '2280766401', '2280767701'],
    'amount': [30, 90, 35, None, 20, 25, 35],
    'transaction': [20, 27, 35, 33, 18, 20, 35],
    'volume': [18, 90, 35, 150, 20, 250, 35],
}

dataF = spark.createDataFrame(pd.DataFrame(my_dict))
dataF.show()
sc.stop()


'''
+------+------------+-----------+------+
|amount|customer_key|transaction|volume|
+------+------------+-----------+------+
|  30.0|  2280765901|         20|    18|
|  90.0|       66001|         27|    90|
|  35.0|    12942501|         35|    35|
|   NaN|  2280766201|         33|   150|
|  20.0|  2280766301|         18|    20|
| 250.0|  2280766401|         20|   250|
|  35.0|  2280767701|         35|    35|
+------+------------+-----------+------+

La prima cosa che vedo è il nome delle colonne:
'''

print (dataF.columns)


['amount', 'customer_key', 'transaction', 'volume']

'''

•	Massimo & Minimo

Vedremo due metodi 

1.	Creo una nuova colonna array contenente le colonne su cui vuoi calcolare il massimo o minimo in un array
'''

colonne = ['amount', 'volume']
newdf = ll.withColumn('array_columns', F.array(colonne))

'''
+------------+------+-----------+------+-------------+
|customer_key|amount|transaction|volume|array_columns|
+------------+------+-----------+------+-------------+
|  2280765901|  30.0|         20|    18| [30.0, 18.0]|
|       66001|  90.0|         27|    90| [90.0, 90.0]|
|    12942501|  35.0|         35|    35| [35.0, 35.0]|
|  2280766201|   NaN|         33|   150| [NaN, 150.0]|
|  2280766301|  20.0|         18|    20| [20.0, 20.0]|
+------------+------+-----------+------+-------------+


Adesso utilizzo  F.sort_array in modo ascendente e discendente per calcolarmi il massimo e minimo
 '''
          
.withColumn('max', F.sort_array("array_columns", False)[0])\
.withColumn('min', F.sort_array("array_columns", True)[0])

'''
+------------+------+-----------+------+-------------+-----+-----+
|customer_key|amount|transaction|volume|array_columns|  max|  min|
+------------+------+-----------+------+-------------+-----+-----+
|  2280765901|  30.0|         20|    18| [30.0, 18.0]| 30.0| 18.0|
|       66001|  90.0|         27|    90| [90.0, 90.0]| 90.0| 90.0|
|    12942501|  35.0|         35|    35| [35.0, 35.0]| 35.0| 35.0|
|  2280766201|   NaN|         33|   150| [NaN, 150.0]|  NaN|150.0|
|  2280766301|  20.0|         18|    20| [20.0, 20.0]| 20.0| 20.0|


Importante: vedi per i cai null mi calcola come massimo il NaN e come minimo il valore.


2.	Utilizzo  least()  e greatest()
'''

# Secondo Metodo
colonne = ['amount', 'volume']
newdf = ll\
    .withColumn('max', F.greatest(*colonne))\
    .withColumn('min', F.least(*colonne))

'''
Ho messo * perchè cosi spezzo la lista è come se fosse ‘amount’,’volume’


+------------+------+-----------+------+-----+-----+
|customer_key|amount|transaction|volume|  max|  min|
+------------+------+-----------+------+-----+-----+
|  2280765901|  30.0|         20|    18| 30.0| 18.0|
|       66001|  90.0|         27|    90| 90.0| 90.0|
|    12942501|  35.0|         35|    35| 35.0| 35.0|
|  2280766201|   NaN|         33|   150|  NaN|150.0|
|  2280766301|  20.0|         18|    20| 20.0| 20.0|

•	Somma 

Sommo tutte le colonne tranne la customer_key,vedramo diversi metodi:

1.	Creo una nuova colonna array contenente le colonne su cui vuoi somma:
'''

colonne = [col for col in ll.columns if col !='customer_key']
df = ll.withColumn('array_columns', F.array(colonne))\
    .select('*', sum([F.col('array_columns').getItem(i) for i in range(len(colonne))]).alias('Total'))

'''
+------------+------+-----------+------+-------------------+-----+
|customer_key|amount|transaction|volume|      array_columns|Total|
+------------+------+-----------+------+-------------------+-----+
|  2280765901|  30.0|         20|    18| [30.0, 20.0, 18.0]| 68.0|
|       66001|  90.0|         27|    90| [90.0, 27.0, 90.0]|207.0|
|    12942501|  35.0|         35|    35| [35.0, 35.0, 35.0]|105.0|
|  2280766201|   NaN|         33|   150| [NaN, 33.0, 150.0]|  NaN|
|  2280766301|  20.0|         18|    20| [20.0, 18.0, 20.0]| 58.0|


Importante :come vedi la somma con un valore NaN da NaN anche se le altre colonne non sono missing. Devi imputare prima a zero.

2.	Uso l’operatore sum 
'''

newdf = ll.withColumn('total_somma_colonne', sum(ll[col] for col in ll.columns if col !='customer_key'))


'''
Importante: qui non devi mettere F.sum ma sum  ,devi usare la sum di Python e non di Pyspark

+------------+------+-----------+------+-------------------+
|customer_key|amount|transaction|volume|total_somma_colonne|
+------------+------+-----------+------+-------------------+
|  2280765901|  30.0|         20|    18|               68.0|
|       66001|  90.0|         27|    90|              207.0|
|    12942501|  35.0|         35|    35|              105.0|
|  2280766201|   NaN|         33|   150|                NaN|
|  2280766301|  20.0|         18|    20|               58.0|


Importante :come vedi la somma con un valore NaN da NaN anche se le altre colonne non sono missing. Devi imputare prima a zero.


3.	Secondo metodo:
'''

def column_add(a,b):
     return  a.__add__(b)

newdf = dataF.withColumn('total_col', reduce(column_add, ( dataF[col] for col in dataF.columns if col !='customer_key')))


'''
Ho lo stesso risultato di sopra.

Per la sottrazione posso utilizzare il metodo magico:

'''
__sub__  al posto di __add__


'''
4.	Terzo  metodo
Se voglio fare una somma tra due sole colonne posso usare F.col() + F.col() oppure un metodo più elegante che è udf
'''

func = udf(lambda arr: arr[0] + arr[1],FloatType())
df = dataF.withColumn('somma', func(F.array('transaction', 'amount')))

'''
+------+------------+-----------+------+-----+
|amount|customer_key|transaction|volume|somma|
+------+------------+-----------+------+-----+
|  30.0|  2280765901|         20|    18| 50.0|
|  90.0|       66001|         27|    90|117.0|
|  35.0|    12942501|         35|    35| 70.0|
|   NaN|  2280766201|         33|   150|  NaN|
|  20.0|  2280766301|         18|    20| 38.0|
|  25.0|  2280766401|         20|   250| 45.0|
|  35.0|  2280767701|         35|    35| 70.0|
+------+------------+-----------+------+-----+

Importante quando sommi due colonne devi vedere il tipo di ritorno deve essere simile/uguale ai tipi delle due colonne che sommi:

root
 |-- amount: double (nullable = true)
 |-- customer_key: string (nullable = true)
 |-- transaction: long (nullable = true)
 |-- volume: long (nullable = true)

se ad esempio provo a dare alla somma delle due colonne transaction + volume il tipo di ritorno float dell’udf
'''

func2 = udf(lambda arr: arr[0]*arr[1],FloatType())
df2 = dataF.withColumn('somma', func2(F.array('transaction', 'volume')))

'''
+------+------------+-----------+------+-----+
|amount|customer_key|transaction|volume|somma|
+------+------------+-----------+------+-----+
|  30.0|  2280765901|         20|    18| null|
|  90.0|       66001|         27|    90| null|
|  35.0|    12942501|         35|    35| null|
|   NaN|  2280766201|         33|   150| null|
|  20.0|  2280766301|         18|    20| null|
|  25.0|  2280766401|         20|   250| null|
|  35.0|  2280767701|         35|    35| null|
+------+------------+-----------+------+-----+

come vedi sono tutti null i risultati
se invece come visto dal print schema metto Long
'''

func2 = udf(lambda arr: arr[0]*arr[1],LongType())
df2 = dataF.withColumn('somma', func2(F.array('transaction', 'volume')))

'''
+------+------------+-----------+------+-----+
|amount|customer_key|transaction|volume|somma|
+------+------------+-----------+------+-----+
|  30.0|  2280765901|         20|    18|  360|
|  90.0|       66001|         27|    90| 2430|
|  35.0|    12942501|         35|    35| 1225|
|   NaN|  2280766201|         33|   150| 4950|
|  20.0|  2280766301|         18|    20|  360|
|  25.0|  2280766401|         20|   250| 5000|
|  35.0|  2280767701|         35|    35| 1225|
+------+------------+-----------+------+-----+

Stai attento al tipo ,come vedi adesso è corretto
'''
'''

•	Moltiplicazione

1.	Primo metodo:

Al momento non ho trovato un metodo uguale a quello della somma.

2.	Secondo metodo:

utilizzo il metodo magico mul ,che è la moltiplicazione:
'''

def column_molt (a,b):
     return  a.__mul__(b)

newdf = dataF.withColumn('total_col', reduce(column_molt, ( dataF[col] for col in dataF.columns if col !='customer_key')))

'''
+------+------------+-----------+------+---------+
|amount|customer_key|transaction|volume|total_col|
+------+------------+-----------+------+---------+
|  30.0|  2280765901|         20|    18|  10800.0|
|  90.0|       66001|         27|    90| 218700.0|
|  35.0|    12942501|         35|    35|  42875.0|
|   NaN|  2280766201|         33|   150|      NaN|
|  20.0|  2280766301|         18|    20|   7200.0|
| 250.0|  2280766401|         20|   250|1250000.0|
|  35.0|  2280767701|         35|    35|  42875.0|
+------+------------+-----------+------+---------+
'''

'''
•	Media

1.	Primo metodo:
'''

marksColumns = [F.col('age3'), F.col('age5')]
averageFunc = sum(x for x in marksColumns)/len(marksColumns)
dataF = dataF.withColumn('avg', averageFunc)


 


# •	Altro metodo con udf ,da testare

from pyspark.sql.functions import udf, array
from pyspark.sql.types import DoubleType

avg_cols = udf(lambda array: sum(array)/len(array), DoubleType())

df.withColumn("average", avg_cols(array("marks1", "marks2"))).show()


'''
3.Terzo  metodo

come la somma puii suare udf per due o tre colonne o più colonne
'''
func = udf(lambda arr: arr[0] * arr[1],FloatType())
df = dataF.withColumn('moltiplicazione', func(F.array('transaction', 'amount')))

'''
anche qui valgono le regole dei tipi visto nella somma
'''