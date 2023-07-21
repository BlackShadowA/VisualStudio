import pandas as pd
import py_mob as ll

df = pd.read_excel("C:\\Travaux_2012\\Esempi_python\\ccdata.xls")
print(df)

y = df['default payment next month']
utl = df['BILL_AMT5']

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