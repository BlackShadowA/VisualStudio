import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_excel("C:\\Travaux_2012\\Esempi_python\\dataset\\ccdata.xls")
print(df)

y = df['default payment next month']
df = df.drop('default payment next month', axis = 1)  

#Per adesso elimino le colonne categoriche
column_string = [ col  for col, dt in df.dtypes.items() if dt == object]
print(column_string)
x = df.drop(column_string, axis = 1)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)



hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['AUC'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,
    "max_bin": 512,
    "num_iterations": 100
}

gbm = lgb.LGBMClassifier(**hyper_params)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='AUC',
        early_stopping_rounds=1000)


feature_imp = pd.DataFrame({'Value':gbm.feature_importances_,'Feature':X_train.columns})
plt.figure(figsize=(20, 10))
sns.set(font_scale = 2)
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:20])
plt.title('LightGBM Features Importance')
plt.tight_layout()
plt.show()