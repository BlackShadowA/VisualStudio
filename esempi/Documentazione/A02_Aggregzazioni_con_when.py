
def my_compute_function(my_input):

    agg_mu = F.sum(F.when(F.col('tipo_prodotto') == 'MU', F.col('capitale_residuo'))\
                    .otherwise(0)).alias('capitale_residuo_mutui')
    agg_pp = F.sum(F.when(F.col('tipo_prodotto') == 'PP', F.col('capitale_residuo'))\
                    .otherwise(0)).alias('capitale_residuo_prestiti')
    aggregation = my_input.groupby('ndg_s').agg(agg_mu, agg_pp)

    return aggregation

#Esempio generale:
#ho il seguente dataframes:

+---+---+----+---+
|  x| id|   y|  z|
+---+---+----+---+
|  a|  1|2502|332|
|  b|  1|2328| 56|
|  a|  1|  21| 78|
|  b|  2| 234| 23|
|  b|  2| 785| 12|
+---+---+----+---+

#Aggrego con condizione sulla colonna x

df_new = df.groupBy("id").agg(
                        F.avg(F.when((F.col("x") == 'a'), F.col('y')).otherwise(0)).alias('col1'),
                        F.count(F.when((F.col("x") == 'b'), F.col('y'))).alias('col2'),
                        F.sum(F.when((F.col("x") == 'b'), F.col('y')).otherwise(0)).alias('col3'))

df_new.show()


+---+-----+----+----+
| id| col1|col2|col3|
+---+-----+----+----+
|  1|841.0|   1|2328|
|  2|  0.0|   2|1019|
+---+-----+----+----+

