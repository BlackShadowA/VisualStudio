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