import pandas as pd
import MOB  as ll
from MOB_PLOT import MOB_PLOT

df = pd.read_csv('C:\\Users\\ur00601\\Downloads\\Monotonic-Optimal-Binning-main\\data\\german_data_credit_cat.csv')
df['default'] = df['default'] - 1
print(df)
df = df[['default','Durationinmonth']]
print(df)
MOB_ALGO = ll.MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = 0)
MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                max_samples = 0.4, min_samples = 0.05, 
                                min_bads = 0.05, 
                                init_pvalue = 0.4, 
                                maximize_bins=True)
# mergeMethod = 'Size' means to run MOB algorithm under bins size base
SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
print(SizeBinning)
StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
print(SizeBinning)

# plot the bin summary data.
print('Bins Size Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = 'Durationinmonth')
print('Statisitcal Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = 'Durationinmonth')


