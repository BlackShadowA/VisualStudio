# Semplicemente puoi fare:


per_join = customer_history.filter(customer_history.snapshot_date >= customer_history.agg({
        "snapshot_date": "max"
    }).collect()[0][0])


# oppure con una funzione


def get_latest_snapshot(dataframe, snapshot_column):
        last_snapshot_date = dataframe.agg(F.max(snapshot_column).alias("last_snapshot_date")).first().last_snapshot_date
        return dataframe.filter(F.col(snapshot_column) == last_snapshot_date)

last = get_latest_snapshot(egan, "snapshot_date")



from transforms.api import transform_df, Input, Output
import pyspark.sql.functions as F

@transform_df(
    Output("/uci/hidden_affluent_campaign/bank_organization_maxdate"),
    tCLP=Input("/uci/dwh_crm/clean/bank_organization_history"),
)
def my_compute_function(tCLP):
    max_date = tCLP.agg(F.max(F.col("snapshot_date")).alias("max")).select("max").collect()[0].max
    return tCLP.filter(F.col("snapshot_date") == max_date)


oppure hai una libreria

from pythonutils import dataframeutils

@transform_df(
    Output("/uci/ontology/data/pipeline/risalita_commerciale/branch"),
    tgtb111r=Input("/uci/policy/derived/tgtb111r
)
def compute(tgtb111r, anagrafica_agenzia_normalized, tgtb0204, policy_df):
    tgtb111r = dtgtb111r.get_latest_snapshot(dtgtb111r, "snapshot_date")

# ecco il codice di get_latest_snapshot



import pyspark.sql.functions as F

def get_latest_snapshot(dataframe, snapshot_column):
    last_snapshot_date = dataframe.agg(F.max(snapshot_column).alias("last_snapshot_date")).first().last_snapshot_date
    return dataframe.filter(F.col(snapshot_column) == last_snapshot_date)
