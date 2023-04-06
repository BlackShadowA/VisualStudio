# Se vuoi crearti una chiave da 1 a n  come chiave.

   
 w = Window().orderBy(F.lit('cust_id'))
 df = df.withColumn('key', F. row_number().over(w).alias("key"))
