expr_max = [F.max(c).alias(c) for c in datamart_controparti_20201231.columns if c.startswith("fl_")]
expr_sum = [F.sum(c).alias(c) for c in datamart_controparti_20201231.columns if c.startswith(("stock”))]

        .groupBy("cust_id")
        .agg(
            *expr_max,
            *expr_sum,
            F.max("branch_structure_cd").alias("branch_structure_cd"),
            F.min("data_apertura_conto_corrente").alias("data_apertura_conto_corrente")
)



# Aggregazione con dizionari
#dizionario
num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

# Altro metodo con rename delle colonne:


num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    #rename
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    

# Pyspark:

aggs = [F.sum('Importance').alias('Importance'),
                F.min('Index').alias('Min_Index'),
                F.max('Index').alias('Max_Index')]
        df = df.groupBy('Column').agg(*aggs)



from pyspark.sql.functions import min

exprs = [min(x) for x in df.columns]
df.groupBy("col1").agg(*exprs).show()



exprs = {x: "sum" for x in df.columns}
df.groupBy("col1").agg(exprs).show()

 


# Con più aggregazione

    aggregates = [
        F.sum("cliente_flag").alias("clienti"),
        F.sum("contattato_flag").alias("contattati"),
        F.sum("interessato_flag").alias("interessati"),
        F.sum("primo_contatto_flag").alias("primi_contatti"),
        F.sum("recall_flag").alias("recalls"),
        F.sum("contatto_da_lavorare_flag").alias("contatti_da_lavorare"),
        F.sum("contatto_finale_flag").alias("contatti_finali"),
        F.sum("appuntamento_scaduto_flag").alias("appuntamenti_scaduti"),
        F.sum("appuntamento_oggi_flag").alias("appuntamenti_oggi"),
        F.sum("appuntamento_domani_flag").alias("appuntamenti_domani"),
        F.sum("appuntamento_last_five_days_flag").alias("appuntamenti_last_five_days"),
        F.sum("appuntamento_next_five_days_flag").alias("appuntamenti_next_five_days"),
        F.sum("interessato_senza_appuntamento").alias("interessati_senza_appuntamento"),
        F.sum("afi_apportate_flag_last_five_days").alias("afi_apportate_last_five_days"),
        F.sum("accettato_flag_last_five_days").alias("accettati_last_five_days"),
        F.sum("interessato_flag_last_five_days").alias("interessati_last_five_days"),
        F.sum("non_interessato_flag_last_five_days").alias("non_interessati_last_ days")
    ]

    commercial_area_statistics = target_customer.groupBy("country_structure_code", "region_structure_code", "commercial_area_structure_code") \
        .agg(*aggregates) \
