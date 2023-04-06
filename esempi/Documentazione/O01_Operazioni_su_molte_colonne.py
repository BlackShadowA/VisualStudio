# se voglio esempio dividere tutte le colonne numeriche per 10 puoi fare direttamente cosi:

selectPctColsExpr = [(F.col(z)/100).cast("decimal(3,2)").alias(z) for z in colWithPct]
selectRemainingColsExpr = [‘ndg_s’]
df.select(selectPctColsExpr+selectAmtColsExpr)

'''
in questo modo dividi le colonne selectPctColsExpr diviso 10:
Posso creare dei flag su molti clienti invece di fare tanti withColumn
'''

variabili_da_far_diventare_flag = ['cust_email_mkt_consent', 'Moratoria_Mutui_Banca']
selectPctColsExpr = [(F.when(F.col(z) == 'S', 'S').otherwise('N').alias('flag_rr_' + z)) for z in variabili_da_far_diventare_flag]
moratoria_128 =cura_italia_individuals.select(selectPctColsExpr)

# Anche questo:

    exprs = []
    cols = ['mth_srs_prod_s0',
        'mth_srs_prod_s1',
        'mth_srs_prod_s2',
        'mth_srs_prod_s3',
        'mth_srs_prod_s4',
        'mth_srs_prod_s5',
        'mth_srs_prod_s6',
        'mth_srs_prod_s7',
        'mth_srs_prod_s8',
        'mth_srs_prod_s9',
        'mth_srs_prod_s10',
        'mth_srs_prod_s11',
        'mth_srs_prod_s12']

    for i in cols:
        exprs.append(F.sum(F.col(i)).alias('sum_' + i))

    df_afi = df_attrib.groupBy(["ndg_key"]).agg(*exprs)      

    return df_afi
