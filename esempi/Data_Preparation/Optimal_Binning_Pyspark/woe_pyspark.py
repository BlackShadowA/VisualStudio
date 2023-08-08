char_vars = []
num_vars = []
final_vars  = []
max_bin = 5
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


def WOE(df, feature, target  = '', n = 5):
    import scipy.stats.stats as stats
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import QuantileDiscretizer
    from pyspark.ml.feature import VectorAssembler
    import scipy.stats.stats as stats
    from pyspark.sql import functions as F
    import pandas as pd
    import numpy as np

    spark = SparkSession.builder.getOrCreate()

    def mono_bin(df, feature, target, n = max_bin):
        r = 0
        while np.abs(r) < 1 and n>1:
            try:
                qds = QuantileDiscretizer(numBuckets=n, inputCol=feature, outputCol="buckets", relativeError=0.01)
                bucketizer = qds.fit(df)
                df = bucketizer.transform(df)
                corr_df = df.groupby('buckets').agg({feature: 'avg', target: 'avg'}).toPandas()
                corr_df.columns = ['buckets', feature, target]
                r, p = stats.spearmanr(corr_df[feature], corr_df[target])
                n = n-1
            except Exception as e:
                n=n-1
            return df

    count = -1
    for feature in feature:
        count = count+1
        if feature!=target:
            temp_df = df.select([feature, target])
            temp_df = mono_bin(temp_df, feature, target, n = max_bin)
            grouped = temp_df.groupby('buckets')
            count_df = grouped.agg(F.count(target).alias('count')).toPandas()
            event_df = grouped.agg(F.sum(target).alias('event')).toPandas()
            min_value = grouped.agg(F.min(feature).alias('min')).toPandas()['min']
            max_value = grouped.agg(F.max(feature).alias('max')).toPandas()['max']
            woe_df = pd.merge(left=count_df, right=event_df)
            woe_df['start'] = min_value
            woe_df['end'] = max_value
            woe_df['non_event'] = woe_df['count'] - woe_df['event']
            woe_df['event_rate'] = woe_df['event']/woe_df['count']
            woe_df['nonevent_rate'] = woe_df['non_event']/woe_df['count']
            woe_df['dist_obs'] = woe_df['count'] / woe_df['count'].sum()
            woe_df['dist_event'] = woe_df['event']/woe_df['event'].sum()
            woe_df['dist_nonevent'] = woe_df['non_event']/woe_df['non_event'].sum()
            woe_df['woe'] = np.log(woe_df['dist_event']/woe_df['dist_nonevent'])
            woe_df['iv'] = (woe_df['dist_event']-woe_df['dist_nonevent'])*woe_df['woe']
            woe_df['iv_grp'] = (woe_df['dist_event'] - woe_df['dist_nonevent']) * woe_df['woe']
            woe_df['varname'] = [feature]* len(woe_df)
            woe_df = woe_df[['varname', 'start', 'end', 'count', 'event', 'non_event', 'event_rate', 'nonevent_rate','dist_obs', 'dist_event', 'dist_nonevent', 'woe', 'iv', 'iv_grp']]
            woe_df = woe_df.replace([np.inf, -np.inf], 0)
            woe_df.iv_grp = woe_df.iv_grp.round(4)
            woe_df['iv'] = woe_df['iv'].sum()
            if count == 0:
                final_woe_df = woe_df
            else:
                final_woe_df = final_woe_df.append(woe_df, ignore_index=True)
            iv = pd.DataFrame({'IV': final_woe_df.groupby('varname').iv.max()})
            iv = iv.reset_index()
    return spark.createDataFrame(final_woe_df), spark.createDataFrame(iv)

