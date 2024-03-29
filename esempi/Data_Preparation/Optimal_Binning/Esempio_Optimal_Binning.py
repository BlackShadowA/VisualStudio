import pandas as pd
import py_mob as ll


df = pd.read_csv('C:\\Users\\ur00601\\Downloads\\Monotonic-Optimal-Binning-main\\data\\german_data_credit_cat.csv', sep=';')
print(df)

df['default'] = df['default'] - 1
print(df)
df = df[['default','Durationinmonth']]

y = df['default']
utl = df['Durationinmonth']

utl_bin = ll.qtl_bin(utl, y)
ll.view_bin(utl_bin)

ll.summ_bin(utl_bin)

ll.view_bin(ll.bad_bin(utl, y))

ll.view_bin(ll.iso_bin(utl, y))

ll.view_bin(ll.rng_bin(utl, y))

ll.view_bin(ll.kmn_bin(utl, y))

'''
rst = ll.pd_bin(df['default payment next month'], df[['BILL_AMT5', 'BILL_AMT6']])
print(rst)

out = ll.pd_woe(df[['ltv', 'bureau_score', 'tot_derog']], rst["bin_out"])
print(out)
'''