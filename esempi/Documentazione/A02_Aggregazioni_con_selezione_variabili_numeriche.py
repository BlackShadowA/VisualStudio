
# Seleziono le variabili numeriche
variabili_numeriche =[nome for nome, tipo in df.dtypes if tipo not in ('string','date')]


# Adesso mi creo un dizionario con nome delle variabili numeriche e :variance per poi metterlo in agg di aggregazione:


# mi calcolo un dizionario per poi mettere in agg    
diz = {}
for nf in variabili_numeriche:
    diz[nf] = 'variance'
print(diz)



{'label': 'variance', 'fl_prestito_in_essere': 'variance', 'fl_prestito_estinto_24m': 'variance', 'fl_prestito_estinto_24m_anticipatamente': 'variance'
