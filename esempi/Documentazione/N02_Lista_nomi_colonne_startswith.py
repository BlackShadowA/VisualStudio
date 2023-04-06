# Se voglio un elenco delle colonne che iniziano per usa questo codice


result = [i for i in df.columns if i.startswith('crif_') |  i.startswith('expirian_')]


# se voglio selezionare le colonne che contengo esempio 'aaa'


x = 'aaa'
drop_column = [i for i in ll.columns if x in i]
print(drop_column)

ll = ll.drop(*drop_column)
ll.show()

'''
Posso utilizzare la regular expression.

Ho il seguente dataFrame:

 

Mi creo due liste in cui in una ho i nomi delle colonne che finiscono in _A ed in un altra quelle che finiscono con _N.
'''

import re


# prima utilizzando le regolar expression prendo quelle che finiscono con _A:


amt = [c for c in df.columns if re.match(r'\w*_A$', c)]

 
 

# poi quelle che finiscono con _N


num = [c for c in df.columns if re.match(r'\w*_N$', c)]


 
