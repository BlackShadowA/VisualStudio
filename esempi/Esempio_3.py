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
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='250' then workday00_bal_vl else 0 end")).alias("stock_gp"),
            F.sum(F.expr("case when mkt_prod_hier_lev03_cd ='270' then workday00_bal_vl else 0 end")).alias("stock_bas"),
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
