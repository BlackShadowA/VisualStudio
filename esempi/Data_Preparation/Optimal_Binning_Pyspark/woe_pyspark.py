from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import VectorAssembler
import scipy.stats.stats as stats
from pyspark.sql import functions as F
import pandas as pd
import numpy as np

char_vars = []
num_vars = []
final_vars  = []
max_bin = 20
custom_rho = 1
fearure = []

def var_type(df):
    vars_list = df.dtypes
    char_vars = []
    num_vars = []
    for i in vars_list:
        if i[1] in ('string', 'date'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars


def calculate_woe(count_df, event_df, min_value, max_value, feature):
    woe_df = pd.merge(left=count_df, right=event_df)
    woe_df['min_value'] = min_value
    woe_df['max_value'] = max_value
    woe_df['non_event'] = woe_df['count'] - woe_df['event']
    woe_df['event_rate'] = woe_df['event']/woe_df['count']
    woe_df['nonevent_rate'] = woe_df['non_event']/woe_df['count']
    woe_df['dist_event'] = woe_df['event']/woe_df['event'].sum()
    woe_df['dist_nonevent'] = woe_df['non_event']/woe_df['non_event'].sum()
    woe_df['woe'] = np.log(woe_df['dist_event']/woe_df['dist_nonevent'])
    woe_df['iv'] = (woe_df['dist_event']-woe_df['dist_nonevent'])*woe_df['woe']
    woe_df['varname'] = [feature]* len(woe_df)
    woe_df = woe_df[['varname', 'min_value', 'max_value', 'count', 'event', 'non_event', 'event_rate', 'nonevent_rate', 'dist_event', 'dist_nonevent', 'woe', 'iv']]
    woe_df = woe_df.replace([np.inf, -np.inf], 0)
    woe_df['iv'] = woe_df['iv'].sum()
    return woe_df

def mono_bin(temp_df, feature, target, n = max_bin):
    r = 0
    while np.abs(r) < custom_rho and n>1:
        try:
            qds = QuantileDiscretizer(numBuckets=n, inputCol=feature, outputCol="buckets", relativeError=0.01)
            bucketizer = qds.fit(temp_df)
            temp_df = bucketizer.transform(temp_df)
            corr_df = temp_df.groupby('buckets').agg({feature: 'avg', target: 'avg'}).toPandas()
            corr_df.columns = ['buckets', fearure, target]
            r, p = stats.spearmanr(corr_df[feature], corr_df[target])
            n = n-1
        except Exception as e:
            n=n-1
        return temp_df

def execute_woe(df, target, final_vars):
    count = -1
    for feature in final_vars:
        if feature!=target:
            count = count+1
            temp_df = df.select([feature, target])
            if feature in num_vars:
                temp_df = mono_bin(temp_df, feature, target, n = max_bin)
                grouped = temp_df.groupby('buckets')
            else:
                grouped = temp_df.groupby(feature)
            count_df = grouped.agg(F.count(target).alias('count')).toPandas()
            event_df = grouped.agg(F.sum(target).alias('event')).toPandas()
            if feature in num_vars:
                min_value = grouped.agg(F.min(feature).alias('min')).toPandas()['min']
                max_value = grouped.agg(F.max(feature).alias('max')).toPandas()['max']
            else:
                min_value = count_df[feature]
                max_value = count_df[feature]
            temp_woe_df = calculate_woe(count_df, event_df, min_value, max_value, feature)
            if count == 0:
                final_woe_df = temp_woe_df
            else:
                final_woe_df = final_woe_df.append(temp_woe_df, ignore_index=True)
        iv = pd.DataFrame({'IV': final_woe_df.groupby('varname').iv.max()})
        iv = iv.reset_index()
    return final_woe_df, iv

