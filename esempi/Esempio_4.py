def runMOB(self, mergeMethod, sign = 'auto') -> pd.DataFrame :
    if mergeMethod in ['Stats', 'Size'] :
            
        outputTable = self.__summarizeBins(FinalOptTable = completeBinningTable)
    
    return outputTable



from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=2)


def columns_with_single_value(df):
    # Calcola il numero di valori unici per ciascuna colonna
    unique_counts = df.agg(*[(F.countDistinct(c).alias(c + '_count')) for c in df.columns])

    # Filtra le colonne con conteggio unico pari a 1
    single_value_columns = [c for c in df.columns if unique_counts.select(col(c + '_count')).first()[0] == 1]

    return single_value_columns

single_value_cols = columns_with_single_value(df)




def feature_selection(iv, trian_test):
    def extract_column_values_to_list(df, column_name):
        values_list = df.select(column_name).rdd.flatMap(lambda x: x).collect()
        return values_list

    df = iv.filter(F.col('IV')>=0.05)

    feature = extract_column_values_to_list(df, 'varname')
    print(feature)

    return trian_test.select('label', *feature)


best_feature = score.idxmax()