from pyspark.sql import functions as F
from transforms.api import transform_df, Input, Output


@transform_df(
    Output("/Users/UR00601/Replatforming/datasets/Prodotti/P001_Conti_Correnti/P001A02_Stock_ContiCorrenti_esplosi"),
    conti_correnti=Input("/uci/ugi-soi/clean/bda0bda0_soi0ccgs"),
    eganag=Input("/uci/ugi-ega/clean/egusqql1_xaaql_"),
    egancm=Input("/uci/ugi-ega/clean/egusqcm1_xaacm_"),
)
def compute(conti_correnti, eganag, egancm):

    lasteganag = eganag.filter(eganag.snapshot_date >= eganag.agg({  # noqa
        "snapshot_date": "max"
    }).collect()[0][0])

    lastegancm = egancm.filter(egancm.snapshot_date >= egancm.agg({  # noqa
        "snapshot_date": "max"
    }).collect()[0][0])

    max_snap = conti_correnti.agg(F.max('SNAPSHOT_DATE')).collect()[0][0]  # noqa

    conti = conti_correnti.filter(F.col('CCTIPO_RAPP_ANAG').isin(['C1', 'C2']))\
                          .filter(F.col('CCDATA_ESTINZIONE').isNull())\
                          .filter(F.col('SNAPSHOT_DATE') == max_snap).dropDuplicates()\
                          .filter(F.col('CCFORMA_TECNICA').isin(
                                ['C2000', 'C2002', 'C2011', 'C2033', 'C2035', 'C2310', 'C2333', 'C2396',
                                 'C2701', 'C2702', 'C2703', 'C2704', 'C2851', 'C2885', 'C2891', 'C2996',
                                 'C2997', 'C2998', 'C2999', 'C3000', 'C3002', 'C3033', 'C3035', 'C3051'
                                 'C3996', 'C3997', 'C3998', 'C3999', 'C5000', 'C5002', 'C5017', 'C5019',
                                 'C5021', 'C5033', 'C5035', 'C5036', 'C5037', 'C5038', 'C5045', 'C5046',
                                 'C5047', 'C5048', 'C5050', 'C5051', 'C5055', 'C5060', 'C5089', 'C5099',
                                 'C5102', 'C5121', 'C5140', 'C5151', 'C5155', 'C5156', 'C5303', 'C5307',
                                 'C5308', 'C5310', 'C5311', 'C5312', 'C5313', 'C5314', 'C5315', 'C5317',
                                 'C5318', 'C5319', 'C5321', 'C5322', 'C5323', 'C5333', 'C5334', 'C5335',
                                 'C5340', 'C5396', 'C5410', 'C5420', 'C5444', 'C5499', 'C5650', 'C5651',
                                 'C5670', 'C5701', 'C5702', 'C5703', 'C5704', 'C5705', 'C5706', 'C5707',
                                 'C5708', 'C5709', 'C5710', 'C5711', 'C5712', 'C5713', 'C5714', 'C5715',
                                 'C5716', 'C5717', 'C5718', 'C5719', 'C5720', 'C5721', 'C5722', 'C5723',
                                 'C5724', 'C5725', 'C5727', 'C5729', 'C5730', 'C5750', 'C5780', 'C5901',
                                 'C5940', 'C5991', 'C5992', 'C5996', 'C5997', 'C5998', 'C5999', 'C8000',
                                 'C8004', 'C8060', 'C8089', 'C8090', 'C8095', 'C8097', 'C8304', 'C8305',
                                 'C8309', 'C8313', 'C9047', 'C9122', 'C9199', 'C9200', 'C9444', 'C9700',
                                 'C9701', 'C9710', 'C9715', 'C9718', 'C9720', 'C9721', 'C9749', 'C9790',
                                 'C5731']))\
                          .withColumn('cntp_id', F.lpad(F.col('CCDIREZ_GENERALE'), 16, '0'))

    conti = conti.join(lasteganag.selectExpr('ndg', 'tipo_ndg'), on=['ndg'])

    lastegancm_co = lastegancm.filter(F.col("cod_colleg") == 'GCO')\
        .selectExpr("ndg as ndg_p", "ndg_collegato as ndg_s", "progr_coint as nr_prog_coint")

    conti = conti\
        .join(lastegancm_co, on=[conti.cntp_id == lastegancm_co.ndg_p],  how="left") \
        .withColumn("ndg_s", F.when(F.col("tipo_ndg") == "CO", F.col("ndg_s")).otherwise(F.col("ndg")))\
        .withColumn('NDG_PAD', F.lpad(F.col('ndg_s'), 16, '0'))

    return conti