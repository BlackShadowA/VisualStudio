# rinominare e mettere in maiuscolsa 
rename_col = [f"{e.upper()}_updated" for e in b.columns]

df = df.toDF(*rename_col)