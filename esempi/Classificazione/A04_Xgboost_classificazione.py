import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_excel("C:\\Travaux_2012\\Esempi_Python\\ccdata.xls")
print(df)

y = df['default payment next month']
df = df.drop('default payment next month', axis = 1)  

#Per adesso elimino le colonne categoriche
column_string = [ col  for col, dt in df.dtypes.items() if dt == object]
print(column_string)
x = df.drop(column_string, axis = 1)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

eval_set = [(X_test, y_test)]

model = xgb.XGBClassifier()
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
