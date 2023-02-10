import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, datetime

date_1 = date(2023, 1, 26)
print(date_1) 

result_1 = date_1 + relativedelta(months=-1)
print(result_1)  

datelist = pd.date_range(result_1, periods=5, freq='W-MON').tolist()
print(datelist)


