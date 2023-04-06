    
def lista_nomi_comuni(a, b):
    result = [i for i in a if i in b]
    return result

colonne_comuni = lista_nomi_comuni(erogato_2022.columns, erogato_2023.columns)



# erogato_2022 e erogato_2023 sono due dataframes Pyspark
