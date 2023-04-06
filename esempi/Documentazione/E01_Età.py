# Calcolo Età

.withColumn('age', F.round(F.datediff(F.current_date(), F.col('date_of_birth')) / 365, 2).cast('integer'))




CASE
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25) < 0 THEN '99 - Others'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 14 THEN '1 - Minorenni 0-13'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 18 THEN '2 - Minorenni 14-17'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 25 THEN '3 - Giovani 18-24'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 35 THEN '4 - Giovani Adulti 25-34'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 45 THEN '5 - Adulti 35-44'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 55 THEN '6 - Adulti 45-54'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 65 THEN '7 - Tardo Adulti 55-64'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  < 75 THEN '8 - Giovani Anziani 65-74'
    WHEN  FLOOR(DATEDIFF(current_date(), date_of_birth) / 365.25)  >= 75 THEN '9 - Anziani 75+'
    ELSE '99 - Others'
END as cust_age_class
