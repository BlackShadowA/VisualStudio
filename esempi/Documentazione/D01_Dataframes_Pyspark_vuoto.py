
from transforms.api import transform_df, Output
from pyspark.sql.types import StructType, StructField, StringType

@transform_df(
    Output("/uci/consumer_finance_campaign/data/apps/bonus_simulator_bk"),
)
def compute(ctx):
    form_id = StructField("bonus_simulator_ID", StringType(), False),
    return ctx.spark_session.createDataFrame([], schema=StructType(form_id))



# con più colonne:

from transforms.api import transform_df, Output
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType, TimestampType

@transform_df(
    Output("/uci/consumer_finance_campaign/data/apps/survey_bk"),
)
def compute(ctx):
    fields = [
        StructField("cliente", StringType(), True),
        StructField("esito_contatto", StringType(), True),
        StructField("dettaglio_contatto", StringType(), True),
        StructField("note", StringType(), True),
        StructField("data_ricontatto", DateType(), True),
        StructField("data_appuntamento", DateType(), True),
        StructField("stato_intermedio", StringType(), True),
        StructField("altro_prodotto_venduto", StringType(), True),
        StructField("importo_prestito", IntegerType(), True),
        StructField("tan", DoubleType(), True),
        StructField("durata_in_anni_prestito", IntegerType(), True),
        StructField("id_esito_contatto", StringType(), False),
        StructField("gestore", StringType(), True),
        StructField("data_contatto", TimestampType(), True),
        StructField("CO_CUSTOMER_TRG", StringType(), True),
        StructField("CO_BANK", StringType(), True),
        StructField("CO-BRANCH", StringType(), True),
        StructField("CO_CAMPAIGN", StringType(), True)
    ]
    return ctx.spark_session.createDataFrame([], schema=StructType(fields))




