# indice di similarità di levenshtein

def similarity(df, col_name_1, col_name_2):

    df = df.withColumn("levenshtein_dist", F.levenshtein(df[col_name_1], df[col_name_2]))

    df = df.withColumn("l1", F.length(df[col_name_1])).withColumn("l2", F.length(df[col_name_2]))
    df = df.withColumn("max", F.when(df["l1"] > df["l2"], df["l1"]).otherwise(df["l2"]))

    df = df.withColumn("similarity", F.lit(1)-(df["levenshtein_dist"]/df["max"]))
    df = df.withColumn("similarity", F.when((df["l1"] == 0) | (df["l2"] == 0), F.lit(0)).otherwise(df["similarity"]))
    df = df.withColumn("levenshtein_dist", F.when((df["l1"] == 0) | (df["l2"] == 0), F.lit(1000)).otherwise(df["levenshtein_dist"]))

    df = df.drop("l1", "l2", "max")

    return df








from transforms.api import Pipeline, Input, Output, transform_df, configure, incremental
import pyspark.sql.functions as F
from pyspark.sql.types import *
import re
import datetime
from pyspark.sql.functions import col, from_unixtime, max as max_, rand
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.ml.feature import RegexTokenizer


tab_name ="counterparty"

@configure(profile=['DRIVER_MEMORY_MEDIUM', 'DRIVER_MEMORY_OVERHEAD_LARGE', 'DRIVER_MEMORY_MEDIUM', 'NUM_EXECUTORS_64', 'EXECUTOR_MEMORY_MEDIUM'])
@transform_df(
    Output("/uci/common_layer/enforced/lev3/transforms-python/src/myproject/datasets/counterparty_x27_0", sever_permissions=True),
    input_bopart_clear=Input("/uci/ugi-bb0/clean_restricted/gold/bb00bda0_bopart___transcode"),
    input_bopart=Input("/uci/ugi-bb0/clean/bb00bda0_bopart__"),
    input_bopart_w_clear=Input("/uci/ugi-bb0/clean_restricted/gold/bb00bda0_bopart_w_transcode"),
    input_bopart_w=Input("/uci/ugi-bb0/clean/bb00bda0_bopart_w"),
    input_boniarri=Input("/uci/ugi-f13/clean/fd13bda0_boniarri"),
    input_boniarri_clear=Input("/uci/ugi-f13/clean_restricted/gold/fd13bda0_boniarri_transcode"),
    input_bonipart=Input("/uci/ugi-f13/clean/fd13bda0_bonipart"),
    input_bonipart_clear=Input("/uci/ugi-f13/clean_restricted/gold/fd13bda0_bonipart_transcode"),
    input_rule=Input("/uci/common_layer/utilities/fusion_sheet/rule/dataframe/counterparty_RULE"),
)
def counterparty_sel(input_bopart, input_bopart_clear, input_bopart_w, input_bopart_w_clear, input_boniarri, input_boniarri_clear, 
input_bonipart, input_bonipart_clear, input_rule, pk=tab_name):

    # Estrazione rule da fusion sheet
    input_rule = input_rule.withColumn("DATE_REF", F.col("DATE_REF").cast("date"))
    input_rule = input_rule.withColumn("DATE_START", F.col("DATE_START").cast("date"))
    input_rule = input_rule.withColumn("DATE_END", F.col("DATE_END").cast("date"))
    input_rule = input_rule.filter(
        (F.col("DATE_REF") >= F.col("DATE_START")) &
        (F.col("DATE_REF") <= F.col("DATE_END"))
    )
    input_rule = input_rule.filter(input_rule["ENFORCED_OUTPUT_FIELD_NAME"].isin(["multibank_in"]))
    
    # interval
    interval_date_rule = input_rule.filter(input_rule["RULE_NAME"].isin(["multibank_indicator_01"]))
    interval_date = interval_date_rule.agg(F.max(F.col("INPUT_VALUE")).alias("interval_date")).first().interval_date
    interval_date = int(interval_date)

    # list unicredit abi
    abi_rule = input_rule.filter(input_rule["RULE_NAME"].isin(["multibank_indicator_00"]))
    abi_rule = abi_rule.withColumn("INPUT_VALUE", F.lpad(F.col("INPUT_VALUE"), 5, '0'))
    abi_group = [row['INPUT_VALUE'] for row in abi_rule.select(F.col("INPUT_VALUE")).collect()]

    # filter
    now = datetime.now()
    last_nr_days = (now.today() - timedelta(days=interval_date)).strftime(format='%Y-%m-%d')
    
    input_bopart = input_bopart.filter(F.col("CBDT_DATA_ESEC").cast("date") >= F.lit(last_nr_days).cast("date"))
    input_bopart_w = input_bopart_w.filter(F.col("CBDT_DATA_ESEC").cast("date") >= F.lit(last_nr_days).cast("date"))
    input_boniarri = input_boniarri.filter(F.col("C11_DT_ESEC_BO").cast("date") >= F.lit(last_nr_days).cast("date"))
    input_bonipart = input_bonipart.filter(F.col("C10_DT_ESEC_BO").cast("date") >= F.lit(last_nr_days).cast("date"))

    # OUTGOING PAYMENTS --------------------------------------------------------------------------------------------------------------------------------------
    # MONTHLY ------------------------------------------------------------------------------------------------------------------------------------------
    input_bopart = input_bopart.select(
        F.col("CBDT_DATA_CAR"),
        F.col("CBDT_CO_KEY_CAR"),
        F.col("CBDT_STATO"),
        F.col("CBDT_BEN_ABI"),
        F.col("CBDT_CRO").alias("cro")
    ).distinct()
    input_bopart = input_bopart.withColumn("CBDT_BEN_ABI", F.lpad(F.col("CBDT_BEN_ABI"), 5, '0'))
    input_bopart = input_bopart.filter(F.col("CBDT_BEN_ABI").isin(abi_group) == False).drop("CBDT_BEN_ABI")
    input_bopart = input_bopart.filter(F.col("CBDT_STATO")==F.lit(1)).drop("CBDT_STATO")

    input_bopart_clear = input_bopart_clear.select(
        F.col("CBDT_DATA_CAR"),
        F.col("CBDT_CO_KEY_CAR"),
        F.col("CBDT_ORD_NDG").alias("ord_ndg"),
        F.col("CBDT_BEN_ANAG_clear").alias("ord_nm"),
        F.col("CBDT_BEN_NDG").alias("ben_ndg"),
        F.col("CBDT_ORD_ANAG_clear").alias("ben_nm")
    ).distinct()
    input_bopart_clear = input_bopart_clear.withColumn("ord_ndg", F.lpad(F.col("ord_ndg"), 16, '0'))
    input_bopart_clear = input_bopart_clear.withColumn("ben_ndg", F.lpad(F.col("ben_ndg"), 16, '0'))
    input_bopart_clear = input_bopart_clear.join(input_bopart, on=["CBDT_DATA_CAR", "CBDT_CO_KEY_CAR"], how="inner")
    input_bopart_clear = input_bopart_clear.withColumn("source", F.lit("M"))

    # OUTGOING PAYMENTS --------------------------------------------------------------------------------------------------------------------------------------
    # WEEKLY ---------------------------------------------------------------------------------------------------------------------
    input_bopart_w = input_bopart_w.select(
        F.col("CBDT_DATA_CAR"),
        F.col("CBDT_CO_KEY_CAR"),
        F.col("CBDT_STATO"),
        F.col("CBDT_BEN_ABI"),
        F.col("CBDT_CRO").alias("cro")
    ).distinct()
    input_bopart_w = input_bopart_w.withColumn("CBDT_BEN_ABI", F.lpad(F.col("CBDT_BEN_ABI"), 5, '0'))
    input_bopart_w = input_bopart_w.filter(F.col("CBDT_BEN_ABI").isin(abi_group) == False).drop("CBDT_BEN_ABI")
    input_bopart_w = input_bopart_w.filter(F.col("CBDT_STATO")==F.lit(1)).drop("CBDT_STATO")

    input_bopart_w_clear = input_bopart_w_clear.select(
        F.col("CBDT_DATA_CAR"),
        F.col("CBDT_CO_KEY_CAR"),
        F.col("CBDT_ORD_NDG").alias("ord_ndg"),
        F.col("CBDT_BEN_ANAG_clear").alias("ben_nm"),
        F.col("CBDT_BEN_NDG").alias("ben_ndg"),
        F.col("CBDT_ORD_ANAG_clear").alias("ord_nm")
    ).distinct()
    input_bopart_w_clear = input_bopart_w_clear.withColumn("ord_ndg", F.lpad(F.col("ord_ndg"), 16, '0'))
    input_bopart_w_clear = input_bopart_w_clear.withColumn("ben_ndg", F.lpad(F.col("ben_ndg"), 16, '0'))
    input_bopart_w_clear = input_bopart_w_clear.join(input_bopart_w, on=["CBDT_DATA_CAR", "CBDT_CO_KEY_CAR"], how="inner")
    input_bopart_w_clear = input_bopart_w_clear.withColumn("source", F.lit("W"))

    outgoing_payments = input_bopart_clear.unionByName(input_bopart_w_clear)
    
    window = Window.partitionBy("CBDT_DATA_CAR", "CBDT_CO_KEY_CAR")

    outgoing_payments = outgoing_payments.withColumn('rank', F.dense_rank().over(window.orderBy(F.col('source').asc())))
    outgoing_payments = outgoing_payments.filter(F.col('rank') == 1).drop("rank", "CBDT_DATA_CAR", "CBDT_CO_KEY_CAR", "source")

    outgoing_payments = outgoing_payments.select(
        F.col("ord_ndg"),
        F.col("ben_ndg"),
        F.col("cro"),
        F.col("ord_nm"),
        F.col("ben_nm"),
        F.lit("outgoing_payments").alias("source")
    ).distinct()

    outgoing_payments = outgoing_payments.withColumn("ndg_not_valid", F.when(
        F.col("ord_ndg").isNull(), F.lit(1)).otherwise(
            F.when(F.col("ord_ndg")==F.lit("0"),F.lit(1)).otherwise(F.lit(0))
        ))
    outgoing_payments = outgoing_payments.filter(F.col("ndg_not_valid")==F.lit(0)).drop("ndg_not_valid")

    # INCOMING SEPA PAYMENTS -----------------------------------------------------------------------------------------------
    input_boniarri = input_boniarri.select(
        F.col("C11_CO_RIF_BO_SEPA").alias("CO_RIF_BO_SEPA"),
        F.col("C11_FL_ST_OPE_BO_SEPA").alias("FL_ST_OPE_BO_SEPA"),
        F.col("C11_CO_IBAN_ORD").alias("CO_IBAN_ORD")).distinct()
    input_boniarri = input_boniarri.withColumn("abi_ord", F.when(
        F.substring(F.col("CO_IBAN_ORD"), 1, 2)==F.lit("IT"), F.substring(F.col("CO_IBAN_ORD"), 6, 5)).otherwise(
            F.lit('00000'))).drop("CO_IBAN_ORD")
    input_boniarri = input_boniarri.filter(F.col("abi_ord").isin(abi_group) == False).drop("abi_ben")
    input_boniarri = input_boniarri.filter(F.col("FL_ST_OPE_BO_SEPA")==F.lit("CO")).drop("FL_ST_OPE_BO_SEPA")    

    input_boniarri_clear = input_boniarri_clear.select(
        F.col("C11_CO_RIF_BO_SEPA").alias("CO_RIF_BO_SEPA"),
        F.col("C11_NDG_ORD").alias("ben_ndg"),
        F.col("C11_DL_ORD_EFF_SEPA_clear").alias("DL_ORD_EFF_SEPA_clear"),
        F.col("C11_DL_ORD_SEPA_clear").alias("DL_ORD_SEPA_clear"),
        F.col("C11_NDG_BEN").alias("ord_ndg"),
        F.col("C11_DL_BEN_EFF_SEPA_clear").alias("DL_BEN_EFF_SEPA_clear"),
        F.col("C11_DL_BEN_SEPA_clear").alias("DL_BEN_SEPA_clear")).distinct()
    input_boniarri_clear = input_boniarri_clear.withColumn("ord_nm", F.when(
        (F.trim(F.col("DL_ORD_EFF_SEPA_clear")).isNull()), F.col("DL_ORD_SEPA_clear")).otherwise(
            F.when(F.trim(F.col("DL_ORD_EFF_SEPA_clear"))==F.lit(""), F.col("DL_ORD_SEPA_clear")).otherwise(
                F.when(F.trim(F.col("DL_ORD_EFF_SEPA_clear"))==F.lit(" "), F.col("DL_ORD_SEPA_clear")).otherwise(F.col("DL_ORD_EFF_SEPA_clear")))#F.col("DL_ORD_SEPA_clear")
        )).drop("DL_ORD_EFF_SEPA_clear", "DL_ORD_SEPA_clear")
    input_boniarri_clear = input_boniarri_clear.withColumn("ben_nm", F.when(
        (F.trim(F.col("DL_BEN_EFF_SEPA_clear")).isNull()), F.col("DL_BEN_SEPA_clear")).otherwise(
            F.when(F.trim(F.col("DL_BEN_EFF_SEPA_clear"))==F.lit(""), F.col("DL_BEN_SEPA_clear")).otherwise(
                F.when(F.trim(F.col("DL_BEN_EFF_SEPA_clear"))==F.lit(" "), F.col("DL_BEN_SEPA_clear")).otherwise(F.col("DL_BEN_EFF_SEPA_clear")))#F.col("DL_ORD_SEPA_clear")
        )).drop("DL_BEN_EFF_SEPA_clear", "DL_BEN_SEPA_clear")

    input_boniarri_clear = input_boniarri_clear.withColumn("ndg_not_valid", F.when(
        F.col("ord_ndg").isNull(), F.lit(1)).otherwise(
            F.when(F.col("ord_ndg")==F.lit("0"),F.lit(1)).otherwise(F.lit(0))
        ))
    input_boniarri_clear = input_boniarri_clear.filter(F.col("ndg_not_valid")==F.lit(0)).drop("ndg_not_valid")

    incoming_sepa_payments = input_boniarri_clear.join(input_boniarri, on=["CO_RIF_BO_SEPA"], how="inner")
    incoming_sepa_payments = incoming_sepa_payments.withColumnRenamed("CO_RIF_BO_SEPA", "cro")
    incoming_sepa_payments = incoming_sepa_payments.withColumn("ord_ndg", F.lpad(F.col("ord_ndg"), 16, '0'))
    incoming_sepa_payments = incoming_sepa_payments.withColumn("ben_ndg", F.lpad(F.col("ben_ndg"), 16, '0'))
    incoming_sepa_payments = incoming_sepa_payments.select(
        F.col("ord_ndg"),
        F.col("ben_ndg"),
        F.col("cro"),
        F.col("ord_nm"),
        F.col("ben_nm"),
        F.lit("incoming_sepa_payments").alias("source")
    ).distinct()

    # OUTGOING SEPA PAYMENTS ---------------------------------------------------------------------------------------------------------------------------------
    input_bonipart = input_bonipart.select(
        F.col("C10_CO_RIF_BO_SEPA").alias("CO_RIF_BO_SEPA"),
        F.col("C10_FL_ST_OPE_BO_SEPA").alias("FL_ST_OPE_BO_SEPA"),
        F.col("C10_CO_IBAN_BEN").alias("CO_IBAN_BEN")).distinct()
    input_bonipart = input_bonipart.withColumn("abi_ben", F.when(
        F.substring(F.col("CO_IBAN_BEN"), 1, 2)==F.lit("IT"), F.substring(F.col("CO_IBAN_BEN"), 6, 5)).otherwise(
            F.lit('00000'))).drop("CO_IBAN_BEN")
    input_bonipart = input_bonipart.filter(F.col("abi_ben").isin(abi_group) == False).drop("abi_ben")
    input_bonipart = input_bonipart.filter(F.col("FL_ST_OPE_BO_SEPA")==F.lit("CO")).drop("FL_ST_OPE_BO_SEPA")    

    input_bonipart_clear = input_bonipart_clear.select(
        F.col("C10_CO_RIF_BO_SEPA").alias("CO_RIF_BO_SEPA"),
        F.col("C10_NDG_ORD").alias("ord_ndg"),
        F.col("C10_NDG_BEN").alias("ben_ndg"),
        F.col("C10_DL_ORD_EFF_SEPA_clear").alias("DL_ORD_EFF_SEPA_clear"),
        F.col("C10_DL_ORD_SEPA_clear").alias("DL_ORD_SEPA_clear"),
        F.col("C10_DL_BEN_SEPA_clear").alias("ben_nm")).distinct()
    input_bonipart_clear = input_bonipart_clear.withColumn("ord_nm", F.when(
        (F.trim(F.col("DL_ORD_EFF_SEPA_clear")).isNull()), F.col("DL_ORD_SEPA_clear")).otherwise(
            F.when(F.trim(F.col("DL_ORD_EFF_SEPA_clear"))==F.lit(""), F.col("DL_ORD_SEPA_clear")).otherwise(
                F.when(F.trim(F.col("DL_ORD_EFF_SEPA_clear"))==F.lit(" "), F.col("DL_ORD_SEPA_clear")).otherwise(F.col("DL_ORD_EFF_SEPA_clear")))#F.col("DL_ORD_SEPA_clear")
        )).drop("DL_ORD_EFF_SEPA_clear", "DL_ORD_SEPA_clear")

    input_bonipart_clear = input_bonipart_clear.withColumn("ndg_not_valid", F.when(
        F.col("ord_ndg").isNull(), F.lit(1)).otherwise(
            F.when(F.col("ord_ndg")==F.lit("0"),F.lit(1)).otherwise(F.lit(0))
        ))
    input_bonipart_clear = input_bonipart_clear.filter(F.col("ndg_not_valid")==F.lit(0)).drop("ndg_not_valid")

    outgoing_sepa_payments = input_bonipart_clear.join(input_bonipart, on=["CO_RIF_BO_SEPA"], how="inner")
    outgoing_sepa_payments = outgoing_sepa_payments.withColumnRenamed("CO_RIF_BO_SEPA", "cro")
    outgoing_sepa_payments = outgoing_sepa_payments.withColumn("ord_ndg", F.lpad(F.col("ord_ndg"), 16, '0'))
    outgoing_sepa_payments = outgoing_sepa_payments.withColumn("ben_ndg", F.lpad(F.col("ben_ndg"), 16, '0'))
    
    outgoing_sepa_payments = outgoing_sepa_payments.select(
        F.col("ord_ndg"),
        F.col("ben_ndg"),
        F.col("cro"),
        F.col("ord_nm"),
        F.col("ben_nm"),
        F.lit("outgoing_sepa_payments").alias("source")
    ).distinct()

    payments = outgoing_payments.repartition(200).unionByName(outgoing_sepa_payments.repartition(200)).unionByName(incoming_sepa_payments.repartition(200)).distinct()
    
    # filter: only transfers with available ben name and ord name (it means that i'm not considering employees - gold perimeter)
    payments = payments.withColumn("null_value", F.when(
        (F.col("ord_nm").isNull()) | (F.col("ord_nm")==F.lit("")) | (F.col("ord_nm")==F.lit(" ")), F.lit(1)).otherwise(
            F.when((F.col("ben_nm").isNull()) | (F.col("ben_nm")==F.lit("")) | (F.col("ben_nm")==F.lit(" ")), F.lit(1)).otherwise(F.lit(0))
        ))
    payments = payments.filter(F.col("null_value")==F.lit(0))
    
    payments = payments.select(
        F.col("ord_ndg"),
        F.col("ord_nm"),
        F.col("ben_nm"),
        F.col("source")
    ).distinct()

    return payments