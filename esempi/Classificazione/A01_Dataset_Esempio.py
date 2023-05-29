import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

df = pd.read_excel("C:\\Travaux_2012\\Esempi_python\\ccdata.xls")
print(df)

y = df['default payment next month']
df = df.drop('default payment next month', axis = 1)  
print(df)



        

def ll(colonna):
    if colonna>10:
        return 'aa'
    else:
        return'bb'
  
        
df['aa'] = df['ID'].apply(ll)
print(df)