
import pandas as pd
ss = pd.date_range(start="2018-09-09",end="2020-02-02").to_list()
print(ss)

#metodo elegante giorni
start = datetime.datetime.strptime("21-06-2014", "%d-%m-%Y")
end = datetime.datetime.strptime("07-07-2014", "%d-%m-%Y")
date_generated = [((start + datetime.timedelta(days=x)).date().isoformat()) for x in range(0, (end-start).days)]

#metodo elegante in mesi
from dateutil.relativedelta import *
start = datetime.date(2021,12,10)
end = datetime.date(2022,5,10)
periods = relativedelta(end, start).months
date_generated = [((start + relativedelta(months=x)).isoformat()) for x in range(0, periods)]

# mesi
import datetime
from dateutil.relativedelta import *
# from dateutil import relativedelta

start = datetime.date(2021,12,10)
end = datetime.date(2022,5,10)
periods = relativedelta(end, start).months
daterange = []
for i in range(periods):
  date = (start + relativedelta(months=i)).isoformat()
  daterange.append(date)
print(daterange)


# giorni
import datetime
from dateutil.relativedelta import *
start = datetime.date(2021,12,10)
end = datetime.date(2022,5,10)
periods = (end - start).days

daterange = []
for i in range(periods):
    date = (start + datetime.timedelta(days = i)).isoformat()
    daterange.append(date)
print(daterange)



