'''
Nella maggior parte dei casi devi calcolarti delle medie ed hai tante colonne ,naturalmente deiv avere una colonna con gli ammontari ed una con le occorrenze(n)

Mi creo due liste in cui in una ho i nomi delle colonne che finiscono in _A(ammontari) ed in un altra quelle che finiscono con _N(occoreenze)
'''

import re

# prima utilizzando le regolar expression prendo quelle che finiscono con _A:


amt = [c for c in df.columns if re.match(r'\w*_A$', c)]

# poi quelle che finiscono con _N


num = [c for c in df.columns if re.match(r'\w*_N$', c)]

# Adesso mi cacolo le medie con un ciclo di for


for_avg = list(set(num_rid).intersection(amt_rid))
for c in for_avg:
    df = df.withColumn(c+'_AVG', F.col(c+'_A')/F.col(c+'_N'))
 
df.show()
