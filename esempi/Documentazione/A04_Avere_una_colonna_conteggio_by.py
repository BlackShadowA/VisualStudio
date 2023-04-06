#Se vuoi contare i doppi e mantenere tutte le colonne e crearti quindi una sola colonna che ti da il numero di doppi:

w = Window.partitionBy('dl_motiv_bo')  # duplicati di causale
df = (
    boniarri
    .withColumn('count_dup', F.count('dl_motiv_bo').over(w))
)
