import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import numpy as np
import traitlets
import IPython
import ipywidgets
import plotly

df = pd.read_csv("C:\\Travaux_2012\\Esempi_python\\train.csv")
print(df)

y = df["SalePrice"]
x = df.drop('SalePrice',axis = 1)
column_string = [ col  for col, dt in df.dtypes.items() if dt == object]
print(column_string)
x = x.drop(column_string, axis = 1)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)


# CatBoostClassifier
SEED = 1
params = {'loss_function':'RMSE', # objective function
        'eval_metric':'MAE', # metric
        'verbose': 200, # output to stdout info about training process every 200 iterations
        'random_seed': SEED,
        'iterations' : 99999
        }
cbc_1 = CatBoostRegressor(**params)
cbc_1.fit(X_train, y_train, 
        eval_set=(X_test, y_test),
        use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
        plot=True, # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
        early_stopping_rounds=100)