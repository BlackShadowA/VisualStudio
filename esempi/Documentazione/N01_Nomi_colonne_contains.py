# Questo esempio l’ho preso dal categorizzatore ho un dataframes con queste colonne

['ndg', 'AnnoMese', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_presence_mean_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_mean_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_max_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_min_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_sum_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_presence_sd_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_sd_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_count_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_median_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_cv_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_cv_medain_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_10_perc_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_90_perc_feature_over_12month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_presence_mean_feature_over_6month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_mean_feature_over_6month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_max_feature_over_6month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_min_feature_over_6month', 'TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_sum_feature_over_6month']

# Voglio solo le colonne che finiscono con 12month e solo la sum e il count

def selezione_periodo_temporale(transactional_data_uscite):
    columns_list = []
    for column in transactional_data_uscite.columns:
        if '12month' in column and ('sum' in column or 'count' in column):
            columns_list.append(column)
    
    return(
        transactional_data_uscite
        .filter(F.col('AnnoMese') == '201812')
        .select(['ndg']+columns_list)
    )

# Ad esempio ti seleziona una colonna cosi
#TRANSACTIONAL_DATA_USCITE_assets_and_investments_model_amount_sum_feature_over_12month
#Dove hai 12month e sum
