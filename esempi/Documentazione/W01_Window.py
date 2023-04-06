from pyspark.sql.window import Window
    
windowSpec = Window.partitionBy("ndg_key").orderBy("ndg_key")

    aa = df.select('ndg_key','snapshot_date',F.row_number().over(
    windowSpec).alias("New_Row_Num")).filter(F.col('New_Row_Num') == 1)
    return aa


# Se vuoi usare il descending:


def definitivo_tool(da_tool):
    from pyspark.sql.window import Window
    
    windowSpec = Window.partitionBy('ndg').orderBy(F.col("fg_nr_prog_fido").desc())
    ll = da_tool.withColumn('New_Row_Num',F.row_number().over(windowSpec).alias("New_Row_Num"))\
        .filter(F.col('New_Row_Num') == 1)
    return ll
