# rinominare e mettere in maiuscolsa 
rename_col = [f"{e.upper()}_updated" for e in b.columns]

df = df.toDF(*rename_col)


# voglio rinominare tutte le colonne che finiscono con _sum 

def col_replace(df):
    replacements = {c:c.replace('_sum','') for c in df.columns if '_sum' in c}
    df = df.select([col(c).alias(replacements.get(c, c)) for c in df.columns])
    return df
df = col_replace(df)
df.show()