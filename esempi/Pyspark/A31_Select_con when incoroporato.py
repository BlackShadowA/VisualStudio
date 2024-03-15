.select('cliente', 'codice_fiscale', 'figura_anagrafica', 'eta', 'nuovo_cliente',
        'perimetro_retail', 'portafogli_direct', 'canale',
        'co_cana_l1', 'fl_ps', 'fl_personale', 'fl_alfa',
        F.when(F.col("eta").between(0, 12), 'Under 13 - Kids')
            .when(F.col("eta").between(13, 17), '13-17 - Teens')
            .when(F.col("eta").between(18, 30), '18-30 - Young Adults')
            .when(F.col("eta").between(31, 55), '31-55 - Adults')
            .when(F.col("eta").between(56, 70), '56-70 - Matures')
            .when(F.col("eta").between(71, 80), '71-80 - Seniors')
            .when(F.col("eta").between(81, 90), '81-90 - Seniors')
            .when(F.col("eta") > 90, 'Over 90 - Seniors')
            .otherwise('n.d.').alias("fascia_eta"),
        *srtabstru.columns, 'ndg', 'fl_fgc', 'dt_rif')


        .select(
            "reference_date_m",
            *keys[1:],
            *keys_pricing[1:],
            *list(
                sum(
                    (
                        (col, col + "_detailed")
                        for col in [
                            f"count_{suffix}",
                            f"average_saldo_end_of_month_{suffix}",
                            f"sum_{suffix}",
                            f"sum_speculative_component_{suffix}",
                            f"outflow_bond_next_3_months_{suffix}",
                            f"outflow_predicted_{suffix}",
                        ]
                    ),
                    (),
                )
            ),
            f"median_ratio_outflow_next_3_months_over_speculative_{suffix}",
        )
        .distinct()
    )


# Dicotomica
        .select('cust_id',
                F.expr("tp_user_olb_trim in ('D', 'I')").cast('integer').alias('90_days_active_digital_users'),
                F.expr("tp_user_dsk_trim in ('D', 'I')").cast('integer').alias('90_days_active_online_users'),
                F.concat("tp_user_aps_trim", "tp_user_apt_trim", "tp_user_msi_trim", "tp_user_bbk_trim").rlike(r'(D|I)').cast('integer').alias('90_days_active_mobile_users'),
                F.concat("tp_user_aps_trim", "tp_user_bbk_trim").rlike(r'(D|I)').cast('integer').alias('90_days_active_app_users'),
                F.expr("tp_user_olb in ('D', 'I')").cast('integer').alias('30_days_active_digital_users'),
                F.expr("tp_user_dsk in ('D', 'I')").cast('integer').alias('30_days_active_online_users'),
                F.concat("tp_user_aps", "tp_user_apt", "tp_user_msi", "tp_user_bbk").rlike(r'(D|I)').cast('integer').alias('30_days_active_mobile_users'),
                F.concat("tp_user_aps", "tp_user_bbk").rlike(r'(D|I)').cast('integer').alias('30_days_active_app_users'),
            )
        
        
# Nota non la puoi utilizzare con selectExpr