import umap
import pandas as pd

df = pd.read_excel("C:\\Travaux_2012\\Esempi_python\\ccdata.xls")
print(df)
df = df.drop('default payment next month', axis = 1)  

#Per adesso elimino le colonne categoriche
column_string = [ col  for col, dt in df.dtypes.items() if dt == object]
print(column_string)
df = df.drop(column_string, axis = 1)



umap_model = umap.UMAP(metric = "jaccard",
                    n_neighbors = 25,
                    n_components = 2,
                    low_memory = False,
                    min_dist = 0.001)
X_umap = umap_model.fit_transform(df)
df["UMAP_0"], df["UMAP_1"] = X_umap[:,0], X_umap[:,1]
print(df)

# ci mette un pò a girare
