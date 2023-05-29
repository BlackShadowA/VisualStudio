import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Travaux_2012\\Esempi_python\\train.csv")
print(df)

y = df["SalePrice"]
x = df.drop('SalePrice',axis = 1)
column_string = [ col  for col, dt in df.dtypes.items() if dt == object]
print(column_string)
x = x.drop(column_string, axis = 1)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

eval_set = [(X_test, y_test)]

model = xgb.XGBRegressor()
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)