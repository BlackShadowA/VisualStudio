# cols to drop by default from df_traing and df_test
COLS_TO_DROP = ['NDG', 'ndg', 'tax_id', 'cntp_id',  # id
                'reference_date_m', 'reference_date_d',  # ref dates
                'timestamp', 'SNAPSHOT_DATE', 'snapshot_date', 'snapshot_dt',  # custom dates
                'sample', 'in_sample', 'is_insample', # sampling cols
                'cluster', # cluster cols
                'in_time', 'is_intime', 'in_time', 'is_oot', 'oot', # time window cols
                ]

# system const to block the pipeline incresment and conputation overload
__MAX_TRAIN_TEST_PIPE__ = 2
__MAX_APP_PIPE__ = 1