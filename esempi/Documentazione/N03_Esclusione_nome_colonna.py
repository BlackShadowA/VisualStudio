# Un modo elegante per escludere una colonna da un dataframes spark è :


# assume the label column is named "class"
label = "class"

# get a list with feature column names
feature_names = [x.name for x in df.schema if x.name != label]

df = df.select(*feature_names)
