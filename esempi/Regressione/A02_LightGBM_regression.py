import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Travaux_2012\\Esempi_python\\dataset\\train.csv")
print(df)

y = df["SalePrice"]
x = df.drop('SalePrice',axis = 1)
column_string = [ col  for col, dt in df.dtypes.items() if dt == object]
print(column_string)
x = x.drop(column_string, axis = 1)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)


hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,
    "max_bin": 512,
    "num_iterations": 100000
}

gbm = lgb.LGBMRegressor(**hyper_params)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1000)