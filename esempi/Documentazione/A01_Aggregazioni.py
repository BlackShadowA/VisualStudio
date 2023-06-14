'''
Primo metodo voglio aggregare tutte le colonne numeriche :
 
Primo metodo :
'''

carte = [F.sum(c).alias(c) for c in df.columns if c.startswith("n_ope_carte_")]
conto = [F.sum(c).alias(c) for c in df.columns if c.startswith("n_ope_conto_")]

cluster1 = (
    df
    .groupBy("secondario")
    .agg(*carte, *conto)
)

'''
Secondo metodo
'''

column = list(set(input_revenues.schema.names) - set(list_rename))
exprs = [sum_(x).alias(x) for x in column]


agg(*exprs)



'''
Altro metodo:
Voglio sommare tutti I valori delle variabili numeriche by label
'''
from pyspark.sql.functions import sum,col
aggregate = [nome for nome, tipo in df.dtypes if tipo !='string']
groupBy = ["label"]
funs = [sum]
exprs = [f(col(c)).alias(f'{c}_{f.__name__}') for f in funs for c in aggregate]
re1 = df.groupby(*groupBy).agg(*exprs)


''' 
Le variabili come vedi finiscono per _sum
Posso anche mettere più di una aggregazione come metrica ad esempio sum e la media
'''
from pyspark.sql.functions import sum,col,mean
aggregate = [nome for nome, tipo in df.dtypes if tipo !='string']
groupBy = ["label"]
funs = [sum, mean]
exprs = [f(col(c)).alias(f'{c}_{f.__name__}') for f in funs for c in aggregate]
re1 = df.groupby(*groupBy).agg(*exprs)

'''
quelle medie finiscon con _mean
Ecco un esempio in Pysark semplice
'''
import findspark
findspark.init()
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession

spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()

print(f"Versione Pyspark = {spark.version}")

my_dict = {
    'ndg':[10,10,20,30,30],
    'operazioni':[1,2,3,4,5],
    'ammontari':[50,20,10,80,9]
}

ll = spark.createDataFrame(pd.DataFrame(my_dict))

funs = [F.sum, F.count]
cols = ["operazioni", "ammontari"]
aggregazione = ll.groupby('ndg')\
    .agg(*[f(c).alias(f'{c}_{f.__name__}') for c in cols for f in funs])

aggregazione.show()

'''
Adesso vediamo altri esempi
Una cosa vista fare da Prometia ,spesso hai da fare molte aggregazioni insieme ,esempio sum,min,max ecc.In più spesso dei fare delle aggregazioni con dei when .Vedrai due esempi li devi vedere entrambi importantissimo.
Primo esempio più aggregazioni:
ho questo dataframes
'''
df = spark.createDataFrame([(1, 2, 3), (1, 4, 5)], ("x", "y", "z"))
df.show()

'''
+---+---+---+
|  x|  y|  z|
+---+---+---+
|  1|  2|  3|
|  1|  4|  5|
+---+---+---+

Voglio schiacciare per chiave x è crearmi tante operazioni di somma e conteggio ed il massimo , un nuovo modo per farlo è prima mi crevo ,guardalo bene questo un item:
'''
from pyspark.sql import functions as F

funs = [F.sum, F.min, F.max]
cols = ["y", "z"]

print([f(c).alias(f'{c}_{f.__name__}') for c in cols for f in funs])
'''
ecco l’output:

[Column<b'sum(y) AS `y_sum`'>, Column<b'min(y) AS `y_min`'>, Column<b'max(y) AS `y_max`'>, Column<b'sum(z) AS `z_sum`'>, Column<b'min(z) AS `z_min`'>, Column<b'max(z) AS `z_max`'>]

Come vedi il tipo è di questo tipo Column<b'sum(y) AS `y_sum`'>
Questo item lo metto nella funzione di aggregazione:

'''

df = df.groupBy("x")\
       .agg(*[f(c).alias(f'{c}_{f.__name__}') for c in cols for f in funs])

'''
+---+-----+-----+-----+-----+-----+-----+
|  x|y_sum|y_min|y_max|z_sum|z_min|z_max|
+---+-----+-----+-----+-----+-----+-----+
|  1|    6|    2|    4|    8|    3|    5|
+---+-----+-----+-----+-----+-----+-----+


Come vedi ha fatto molte aggregazione .

Secondo esempio più aggregazioni con when:
Adesso ho questo Dataframes
'''
import pandas as pd
my_dict = {
    'customer_key': ['001','001','001','002'],
    'categoria':['regular_incomes','regular_incomes','regular_incomes','occasional_incomes'],
    'richiesto':[10,20,0,5],
    
}
dataF = spark.createDataFrame(pd.DataFrame(my_dict))
dataF.show()

'''
+------------+------------------+---------+
|customer_key|         categoria|richiesto|
+------------+------------------+---------+
|         001|   regular_incomes|       10|
|         001|   regular_incomes|       20|
|         001|   regular_incomes|        0|
|         002|occasional_incomes|        5|
+------------+------------------+---------+
'''

'''
Una cosa che spesso si fa è crearsi tante colonne per le varie categorie (when) .Vediamo passo passo cosa fa Prometei per aggregare.
La prima cosa si crea una lista che contiene le operazioni e le categorie da considerare:
'''

cats_in = ['all', 'regular_incomes', 'occasional_incomes']
cats_out = ['all', 'pippo', 'pluto']
cats_tot = ['all', 'private_money_transfer', ]

filters_in = [(cat, 'in') for cat in cats_in]
filters_out = [(cat, 'out') for cat in cats_out]
filters_tot = [(cat, 'tot') for cat in cats_tot]
filters= filters_in + filters_out + filters_tot 

print(filters)

'''
ecco l’output:
[('all', 'in'), ('regular_incomes', 'in'), ('occasional_incomes', 'in'), ('all', 'out'), ('pippo', 'out'), ('pluto', 'out'), ('all', 'tot'), ('private_money_transfer', 'tot')]
'''

'''
Adesso si creano una serie di funzioni per creare il tipo da inserire in agg.vediamo passo per passo
Prima cosa l’aggregazione che verranno fatte sarannno per i totali le transazioni in entrata e uscita,per questo si creano una prima parte

'''

def sign_filter(c, sign):
    if sign == 'in':
        return F.col(c) > 0
    elif sign == 'out':
        return F.col(c) < 0
    elif sign == 'tot':
        return F.col(c) != 0 #modificato da me per evitare di contare tutto anche gli zero se devi contare anche gli zero metti:  F.lit(True)
    else:
        raise ValueError(f"sign must be one of ['in', 'out', 'tot'] but was {sign}")

def cat_filter(c, cat):
    if cat != 'all':
        return F.col(c) == cat
    else:
        return F.lit(True)

def filters_to_expressions(filters, type_col='categ', value_col='value'):
    return {
        f'{cat}#{sign}': F.when(cat_filter(type_col, cat) & sign_filter(value_col, sign), F.col(value_col))
        for cat, sign in filters
    }

filter_exprs = filters_to_expressions(filters=filters, type_col=type_col, value_col=value_col)

'''
Ecco l’output dela funzione filters_to_expressions:

{'all#in': Column<'CASE WHEN (true AND (richiesto > 0)) THEN richiesto END'>, 'regular_incomes#in': Column<'CASE WHEN ((categoria = regular_incomes) AND (richiesto > 0)) THEN richiesto END'>, 'occasional_incomes#in': Column<'CASE WHEN ((categoria = occasional_incomes) AND (richiesto > 0)) THEN richiesto END'>, 'all#out': Column<'CASE WHEN (true AND (richiesto < 0)) THEN richiesto END'>, 'pippo#out': Column<'CASE WHEN ((categoria = pippo) AND (richiesto < 0)) THEN richiesto END'>, 'pluto#out': Column<'CASE WHEN ((categoria = pluto) AND (richiesto < 0)) THEN richiesto END'>, 'all#tot': Column<'CASE WHEN (true AND (NOT (richiesto = 0))) THEN richiesto END'>, 'private_money_transfer#tot': Column<'CASE WHEN ((categoria = private_money_transfer) AND (NOT (richiesto = 0))) THEN richiesto END'>}

E per ogni categoria e totale 

'''

def round_sum(c):
    return F.round(F.sum(c), 2)

def mean(c):
    return F.round(F.sum(c), 4)/F.count(c)   

def stddev(c):
    return F.sqrt(F.round(F.sum(c**2), 4)*F.count(c) - F.round(F.sum(c), 2)*F.round(F.sum(c), 2))/F.count(c)

def abs_stddev(c):
    return F.stddev(F.abs(c))

def abs_cv(c):
    mean_round = F.round(F.sum(F.abs(c)), 4)/F.count(c)
    return stddev(F.abs(c))*mean_round/(mean_round**2+0.1)

def cv(c):
    mean_round = F.round(F.sum(c), 4)/F.count(c)
    return stddev(c)*mean_round/(mean_round**2+0.1)

agg_funcs = [round_sum, F.count, mean, abs_stddev,abs_cv, F.max,F.min]

agg_exprs = [
        F.round(f(filter_expr), 4).alias(f'{filter_name}#{mult}_{stride}#{f.__name__}')
        for f in agg_funcs
        for filter_name, filter_expr in filter_exprs.items()
    ]


'''
Ecco l’output una parte di agg_exprs:

[Column<'round(round(sum(CASE WHEN (true AND (richiesto > 0)) THEN richiesto END), 2), 4) AS `all#in#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN ((categoria = regular_incomes) AND (richiesto > 0)) THEN richiesto END), 2), 4) AS `regular_incomes#in#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN ((categoria = occasional_incomes) AND (richiesto > 0)) THEN richiesto END), 2), 4) AS `occasional_incomes#in#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN (true AND (richiesto < 0)) THEN richiesto END), 2), 4) AS `all#out#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN ((categoria = pippo) AND (richiesto < 0)) THEN richiesto END), 2), 4) AS `pippo#out#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN ((categoria = pluto) AND (richiesto < 0)) THEN richiesto END), 2), 4) AS `pluto#out#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN (true AND (NOT (richiesto = 0))) THEN richiesto END), 2), 4) AS `all#tot#1_month#round_sum`'>, Column<'round(round(sum(CASE WHEN ((categoria = private_money_transfer) AND (NOT (richiesto = 0))) THEN richiesto END), 2), 4) AS `private_money_


Adesso lo applico al dataframes:
'''
ll = dataF.groupby('customer_key').agg(*agg_exprs)
ll.show()

'''
Ecco una parte  dell’output:
Altro esempio con dizionario:
Seleziono le sole variabili numeriche
'''

# Seleziono le variabili numeriche
# variabili_numeriche =[nome for nome, tipo in df.dtypes if tipo not in ('string','date')]
# Adesso mi creo un dizionario con nome delle variabili numeriche e :variance per poi metterlo in agg di aggregazione:


# mi calcolo un dizionario per poi mettere in agg    
diz = {}
for nf in variabili_numeriche:
    diz[nf] = 'variance'
print(diz)

{'label': 'variance', 'fl_prestito_in_essere': 'variance', 'fl_prestito_estinto_24m': 'variance', 'fl_prestito_estinto_24m_anticipatamente': 'variance'


# Adesso mi calcolo la varianza per tutte le variabili numeriche


# calcolo la variance per tutte le variabili numeriche
varian = df.agg(diz)
varian.show()


 



