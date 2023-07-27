import pandas as pd
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType

df = pd.read_csv('C:\\Travaux_2012\\Esempi_Python\\demo_salary.csv')
print(df)

from upgini import FeaturesEnricher, SearchKey

enricher = FeaturesEnricher(
    search_keys={
    'country': SearchKey.COUNTRY}
    )

train_features = df.drop(['avg_salary'], axis=1)
train_target = df.avg_salary

enriched_train_features = enricher.fit_transform(
    train_features,
    train_target,
    scoring = "mean_absolute_error")