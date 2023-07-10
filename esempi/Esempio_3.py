# verifica

from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("/uci/uprice_investimenti/prioritizzato/datasets/p002_tfa"),
    stock=Input("/uci/cido_consumer/Volumes & Sales/Data/cbk_volume_sales_balance_tot_asset_loans_globe"),
    flussi=Input("/uci/cido_consumer/Volumes & Sales/Data/cbk_volume_sales_movements_tot_asset_globe")
)
def compute(stock, flussi):
    max_snap = stock.agg(F.max("snapshot_date")).collect()[0][0]
    stock_tfa = (
        stock
        .filter(F.col("snapshot_date") == max_snap)
        .filter("mkt_prod_hier_lev02_cd in  ('10','20','25')")
        .groupBy("cntp_id")
        .agg(
            F.sum("workday00_bal_vl").alias("stock_afi"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='10' then workday00_bal_vl else 0 end")).alias("stock_diretta"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='100' then workday00_bal_vl else 0 end")).alias("stock_vista"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='102' then workday00_bal_vl else 0 end")).alias("stock_tempo"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='104' then workday00_bal_vl else 0 end")).alias("stock_obbl_cert_uci"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='20' then workday00_bal_vl else 0 end")).alias("stock_auc"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='200' then workday00_bal_vl else 0 end")).alias("stock_tit_stato_ita"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='202' then workday00_bal_vl else 0 end")).alias("stock_obbligaz"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='245' then workday00_bal_vl else 0 end")).alias("stock_certificates"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='204' then workday00_bal_vl else 0 end")).alias("stock_azioni"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='206' then workday00_bal_vl else 0 end")).alias("stock_warr_cw"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='207' then workday00_bal_vl else 0 end")).alias("stock_etf"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd in ('209','246','208','203')  then workday00_bal_vl else 0 end")).alias("stock_altri_tit"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='25' then workday00_bal_vl else 0 end")).alias("stock_aum"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='250' then workday00_bal_vl else 0 end")).alias("stock_gestione_patrimoniali"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='270' then workday00_bal_vl else 0 end")).alias("stock_bancassurance"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd not in ('250','270') then workday00_bal_vl else 0 end")).alias("stock_fondi"),
            F.sum("prev_year_dec_bal_vl").alias("stock_afi_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='10' then prev_year_dec_bal_vl else 0 end")).alias("stock_diretta_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='100' then prev_year_dec_bal_vl else 0 end")).alias("stock_vista_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='102' then prev_year_dec_bal_vl else 0 end")).alias("stock_tempo_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='104' then prev_year_dec_bal_vl else 0 end")).alias("stock_obbl_cert_uci_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='20' then prev_year_dec_bal_vl else 0 end")).alias("stock_auc_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='200' then prev_year_dec_bal_vl else 0 end")).alias("stock_tit_stato_ita_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='202' then prev_year_dec_bal_vl else 0 end")).alias("stock_obbligaz_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='245' then prev_year_dec_bal_vl else 0 end")).alias("stock_certificates_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='204' then prev_year_dec_bal_vl else 0 end")).alias("stock_azioni_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='206' then prev_year_dec_bal_vl else 0 end")).alias("stock_warr_cw_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='207' then prev_year_dec_bal_vl else 0 end")).alias("stock_etf_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd in ('209',/*no sige*/'246','208','203')  then prev_year_dec_bal_vl else 0 end")).alias("stock_altri_tit_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='25' then prev_year_dec_bal_vl else 0 end")).alias("stock_aum_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='250' then prev_year_dec_bal_vl else 0 end")).alias("stock_gp_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='270' then prev_year_dec_bal_vl else 0 end")).alias("stock_bas_ly"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd not in ('250','270') then prev_year_dec_bal_vl else 0 end")).alias("stock_fondi_ly"),
            F.max("snapshot_date").alias("stock_snapshot_date")
        )
    )
    impieghi = (
        stock
        .filter(F.col("snapshot_date") == max_snap)
        .filter("mkt_prod_hier_lev02_cd in  ('30','31','36','37','38')")
        .groupBy("cntp_id")
        .agg(
            F.sum("workday00_bal_vl").alias("impieghi"),
            F.sum("prev_year_dec_bal_vl").alias("impieghi_ly"),
            F.max("snapshot_date").alias("impieghi_snapshot_date")
        )
    )

    max_snap = flussi.agg(F.max("snapshot_date")).collect()[0][0]
    flussi_tfa = (
        flussi
        .filter(F.col("snapshot_date") == max_snap)
        .filter("mkt_prod_hier_lev02_cd in  ('10','20','25')")
        .groupBy("cntp_id")
        .agg(
            F.sum("curr_year_prog_mov_vl").alias("flussi_afi"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='10' then curr_year_prog_mov_vl else 0 end")).alias("flussi_diretta"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='100' then curr_year_prog_mov_vl else 0 end")).alias("flussi_vista"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='102' then curr_year_prog_mov_vl else 0 end")).alias("flussi_tempo"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='104' then curr_year_prog_mov_vl else 0 end")).alias("flussi_obbl_cert_uci"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='20' then curr_year_prog_mov_vl else 0 end")).alias("flussi_auc"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='200' then curr_year_prog_mov_vl else 0 end")).alias("flussi_tit_stato_ita"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='202' then curr_year_prog_mov_vl else 0 end")).alias("flussi_obbligaz"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='245' then curr_year_prog_mov_vl else 0 end")).alias("flussi_certificates"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='204' then curr_year_prog_mov_vl else 0 end")).alias("flussi_azioni"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='206' then curr_year_prog_mov_vl else 0 end")).alias("flussi_warr_cw"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='207' then curr_year_prog_mov_vl else 0 end")).alias("flussi_etf"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd in ('209',/*no sige*/'246','208','203')  then curr_year_prog_mov_vl else 0 end")).alias("flussi_altri_tit"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='25' then curr_year_prog_mov_vl else 0 end")).alias("flussi_aum"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='250' then curr_year_prog_mov_vl else 0 end")).alias("flussi_gp"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='270' then curr_year_prog_mov_vl else 0 end")).alias("flussi_bas"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd not in ('250','270') then curr_year_prog_mov_vl else 0 end")).alias("flussi_fondi"),
            F.sum(F.expr("case when mkt_prod_hier_lev02_cd ='25' and prenotato_in = 'S' then curr_year_prog_mov_vl else 0 end")).alias("flussi_aum_reg"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='250' and prenotato_in = 'S' then curr_year_prog_mov_vl else 0 end")).alias("flussi_gp_reg"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='270' and prenotato_in = 'S' then curr_year_prog_mov_vl else 0 end")).alias("flussi_bas_reg"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd not in ('250','270')  and prenotato_in = 'S' then curr_year_prog_mov_vl else 0 end")).alias("flussi_fondi_reg"),
            F.max("snapshot_date").alias("flussi_snapshot_date")
        )
    )

    db_final = (
        stock_tfa
        .join(impieghi, 'cntp_id', 'full')
        .join(flussi_tfa, 'cntp_id', 'full')
        .withColumn("vol_sales_snapshot_date", F.coalesce("flussi_snapshot_date", "stock_snapshot_date", "impieghi_snapshot_date"))
        .drop(*["flussi_snapshot_date", "stock_snapshot_date", "impieghi_snapshot_date"])
    )
    return db_final



from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("/Users/UR00601/Replatforming/datasets/Afi_Revenus/A002_Stock_Revenues/A0201_Volumi_revenus"),
    total_revenues_h=Input("/uci/common_layer/enforced/data/total_revenues_h"),
)
def compute(total_revenues_h):
    df = total_revenues_h\
        .filter(F.col('snapshot_dt') == '2022-12-31')\
        .filter(F.col('stru_hie_lev5_de').like('%RETAIL%'))\
        .selectExpr(
            'accprod_hier_lev01_de',
            'accprod_hier_lev02_de',
            'accprod_hier_lev03_de',
            'accprod_hier_lev04_de',
            'accprod_de',
            'tot_rev_cytd_vl as mol_alla_data',
            'tot_rev_fly_vl as mol_fine_anno_mese_precedente',
            'tot_rev_lytd_vl as mol_progressivo_anno_precedente',
            'tot_rev_m0_vl as mol_mese_corrente',
            'tot_avg_lcap_m0_vl as volumi_perido_anno_corrente',
            'tot_avg_lcap_cytd_vl as volumi_progressivo_anno_corrente',
            'tot_avg_lcap_fly_vl as volumi_fine_anno_mese_precedente',
            'tot_avg_lcap_lytd_vl  as volumi_progressivo_anno_precedente'

        )\
        .groupby('accprod_hier_lev01_de',
                 'accprod_hier_lev02_de',
                 'accprod_hier_lev03_de',
                 'accprod_hier_lev04_de',
                 'accprod_de',)\
        .agg(F.sum('mol_alla_data').alias('mol_alla_data'),
             F.sum('volumi_progressivo_anno_corrente').alias('volumi_progressivo_anno_corrente'),
             F.sum('mol_progressivo_anno_precedente').alias('mol_progressivo_anno_precedente'),
             F.sum('mol_fine_anno_mese_precedente').alias('mol_fine_anno_mese_precedente'),
             F.sum('volumi_fine_anno_mese_precedente').alias('volumi_fine_anno_mese_precedente'),
             F.sum('mol_mese_corrente').alias('mol_mese_corrente'),
             F.sum('volumi_perido_anno_corrente').alias('volumi_perido_anno_corrente'),
             F.sum('volumi_progressivo_anno_precedente').alias('volumi_progressivo_anno_precedente'),
        )

    return df



df=Input("/uci/cido_consumer/Volumes & Sales/Data/cbk_volume_sales_balance_tot_asset_loans_globe"),
mutui=Input("/uci/Consumer Finance/analisi_spot/Analisi_Gaetano/Stock_&_Flussi/Stock Mutui/datasets/StockMutui_202212")
)
def compute(df, mutui):
    anno_mese = '2022-12-31'
    last = df.filter(F.col("snapshot_date") <= anno_mese).agg(F.max("snapshot_date")).collect()[0][0]  # noqa
    stock_impieghi = (
        df
        .filter(F.col("snapshot_date") == last)
        .filter("mkt_prod_hier_lev02_cd in  ('30','31','36','37','38')")
        .filter(F.col("workday00_bal_vl") > 0)\
        .groupBy("cntp_id", "macro_area", "mis_deal_nr", "mkt_prod_cd")\
        .agg(
            F.sum("workday00_bal_vl").alias("Impieghi")
 
        )
    )
    mutui = mutui.selectExpr("cntp_id", "rapporto as mis_deal_nr", "co_var_cpc", "im_capres", "im_uti_fi",
                             "dt_contr_fi", "fl_tp_tasso_fi", "flag_prima_casa", "canale_di_vendita")

    stock_impieghi = stock_impieghi.join(mutui, on=["cntp_id", "mis_deal_nr"])

    return stock_impieghi



