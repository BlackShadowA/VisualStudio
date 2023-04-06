# Primo metodo nomi in maiuscolo
df = df.select([F.col(x).alias(x.upper()) for x in df.columns])


# Secondo metodo

df = df.toDF(*[c.lower() for c in df.columns])

