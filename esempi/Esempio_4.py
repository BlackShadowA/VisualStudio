import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import datetime
import foundry_ml
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, precision_score, recall_score,f1_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def cross_validation_kfold(Feature_selection_sigle_Model, Elimino_variabili_con_bassa_Varianza):
    
    # Prediction contribution più è alto il valore più la variabile è importante
    # error_contribution_val se il vaore della variabile è negativa allora non è dannosa per il modello , se positiva si(tende ad aumentare l'errore del modello)
    # Faccio un primo modello prendendo le variabili che hanno un error_contribution_val negativo sul validation set
    lista_variabili =Feature_selection_sigle_Model.filter((F.col('error_contribution') < 0) & (F.col('dataset') == 'validation'))

    def extract_column_values_to_list(df, column_name):
        values_list = df.select(column_name).rdd.flatMap(lambda x: x).collect()
        return values_list

    features = extract_column_values_to_list(lista_variabili, 'index')

    target = "label"
    index = 'customer_key'
    df = cast_decimal_to_double(Elimino_variabili_con_bassa_Varianza) # in lightGBm questo è fondamentale
    colonne_categoriche = [nome for nome, tipo in df.dtypes if tipo =='string']
    df = df.select(target, *features, *colonne_categoriche)

    
    # Divido in Train test e validation
    # Divido il 20 percento (stratificato per la variabile target)
    training , test_set = stratified_split_train_test(dataframes=df, frac=0.8, label=target, join_on=index)
    #df_ = df.join(val_set, on='customer_key', how='left_anti')
    #training, test_set = stratified_split_train_test(dataframes=df_, frac=0.8, label=target, join_on=index)
    
    # Verifica che ho fatto bene
    aa = training.join(test_set, on='customer_key', how='inner')

    print(f"sovrapposizioni = {aa.count()}")


    incidenza_training = incidenza_target(training,  target)
    incidenza_test_set = incidenza_target(test_set,  target)
    print(f"Incidenza variabile target in Training Set = {incidenza_training}, Numero record = {training.count()}")
    print(f"Incidenza variabile target in Test Set = {incidenza_test_set}, Numero record  = {test_set.count()}")

    trainingData = training.toPandas().set_index('customer_key')
    test_set = test_set.toPandas().set_index('customer_key')

    schema = get_schema(trainingData)
    df_train = trainingData.astype(schema, copy=False)
    df_test = test_set.astype(schema, copy=False)
    
    X_train = df_train
    y_train = X_train.pop(target)
    X_test = df_test
    y_test = X_test.pop(target)  
    
    
    def train_model(X, Y, X_test, n_folds = 5):
        kf = KFold(n_splits = n_folds, random_state = 1, shuffle = True)
        risultati = pd.DataFrame()
            
        for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
            # Create data for this fold
            y_train, y_val = Y.iloc[train_index].copy(), Y.iloc[test_index].copy()
            X_train, X_val = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
                        
            
            print( f'Fold: {i}')
            n_features = X_train.shape[1]
            fit_model = lgb.LGBMClassifier(
                class_weight='balanced',
                learning_rate=0.01,
                n_estimators=1000000,
                subsample=1 - np.e**-1,
                subsample_freq=1,
                colsample_bytree=np.sqrt(n_features) / n_features,
                #num_leaves=7,
                importance_type='gain',
            )

            fit_model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric='auc', early_stopping_rounds=200, verbose=100)
            pred_train = fit_model.predict_proba(X_train)[:, 1]
            pred_test = fit_model.predict_proba(X_test)[:, 1]

            y_prediction = fit_model.predict(X_test)

            auc_train = roc_auc_score(y_train, pred_train)
            auc_test = roc_auc_score(y_test, pred_test)
            precision, recall, _ = precision_recall_curve(y_test, pred_test)
            auc_pr = average_precision_score(y_test, pred_test)
            precision = precision_score(y_test, y_prediction)
            recall = recall_score(y_test, y_prediction)
            f1_ = f1_score(y_test, y_prediction)
            fold = f'Fold: {i}'
            model_result = {
                    'Folder': [fold],
                    'AUC_Train' : [auc_train],
                    'AUC_Test' : [auc_test],
                    'AUC-PR': [auc_pr],
                    'Precision' : [precision],
                    'Recall': [recall],
                    'F1': [f1_]  
                }
            app = pd.DataFrame(model_result)
            risultati = risultati.append(app)
            cm = confusion_matrix(y_test, y_prediction)
            print(cm)

        return risultati

    result = train_model(X_train, y_train, X_test)
    print(result)
    
    return spark.createDataFrame(result)


def get_schema(df):
    schema = df.dtypes.apply(lambda x: x.name).to_dict()
    dct2 = {}
    for c, t in schema.items():
        if t == 'object':
            dct2[c] = 'category'
        else:
            dct2[c] = t
    return dct2

def cast_decimal_to_double(df):
    schema = {}
    for col in df.schema:
        schema[col.name] = col.dataType
    cols_new = []
    for c, t in schema.items():
        if str(t).startswith('Decimal'):
            cols_new.append(F.col(c).cast('Double'))
        else:
            cols_new.append(F.col(c))
    return df.select(cols_new)

def stratified_split_train_test(dataframes, frac, label, join_on, seed=42):
    """
    Stratifica
    Parameters:
    - dataframes: PySpark DataFrame
    - frac: % del DataFrame principale
    - label: colonna della variabile Target
    -join_on: la chiave per distinguere i due DataFrame
    Returns:
    - Due dataframes pyspark uno contente la fraction dei dati e uno con 1-fraction
    """
    import pyspark.sql.functions as F
    fractions = dataframes.select(label).distinct().withColumn("fraction", F.lit(frac)).rdd.collectAsMap()
    df_frac = dataframes.stat.sampleBy(label, fractions, seed)
    df_remaining = dataframes.join(df_frac, on=join_on, how="left_anti")
    return df_frac, df_remaining

def incidenza_target(dataframes, variable_target):
    """
    Calcola incidenza della variabile Target
    Parameters:
    - dataframes: PySpark DataFrame
    - variable_target: Name of the variable target
    Returns:
    - Incidence variabile target -- float
    """
    import pyspark.sql.functions as F
    total_rows = dataframes.count()
    target_rows = dataframes.groupby(variable_target).count()
    label = target_rows.filter(F.col(variable_target) == 1).collect()[0][1]
    return (label/total_rows)*100


def shap_sum2proba(shap_sum):
  """Compute sigmoid function of the Shap sum to get predicted probability."""
  
  return 1 / (1 + np.exp(-shap_sum))
def individual_log_loss(y_true, y_pred, eps = 1e-15):
  """Compute log-loss for each individual of the sample."""
  
  y_pred = np.clip(y_pred, eps, 1 - eps)
  return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

def get_preds_shaps(df, features, target):
  """Get predictions (predicted probabilities) and SHAP values for a dataset."""
  X_train = df
  y_train = X_train.pop(target)

  model = LGBMClassifier().fit(X_train, y_train)
  preds = pd.Series(model.predict_proba(X_train)[:,1], index=X_train.index)
  shap_explainer = TreeExplainer(model)
  shap_expected_value = shap_explainer.expected_value[-1]
  shaps = pd.DataFrame(data=shap_explainer.shap_values(X_train)[1],columns=features)
  return preds, shaps, shap_expected_value



def get_feature_contributions(y_true, y_pred, shap_values, shap_expected_value):
  """Compute prediction contribution and error contribution for each feature."""

  prediction_contribution = shap_values.abs().mean().rename("prediction_contribution")
  
  ind_log_loss = individual_log_loss(y_true=y_true, y_pred=y_pred).rename("log_loss")
  y_pred_wo_feature = shap_values.apply(lambda feature: shap_expected_value + shap_values.sum(axis=1) - feature).applymap(shap_sum2proba)
  ind_log_loss_wo_feature = y_pred_wo_feature.apply(lambda feature: individual_log_loss(y_true=y_true, y_pred=feature))
  ind_log_loss_diff = ind_log_loss_wo_feature.apply(lambda feature: ind_log_loss - feature)
  error_contribution = ind_log_loss_diff.mean().rename("error_contribution").T
  
  return prediction_contribution, error_contribution
  

dashboard = Dashboard(
    instances=test_instances,
    local_explanations=local_explanations,
    global_explanations=global_explanations,
    prediction_explanations=prediction_explanations,
    class_names=class_names
)
dashboard.show()
#Dash is running on http://127.0.0.1:8050/



# Creating a list of true values
y_true = [23.5, 45.1, 34.7, 29.8, 48.3, 56.4, 21.2, 33.5, 39.8, 41.6,
          27.4, 36.7, 45.9, 50.3, 31.6, 28.9, 42.7, 37.8, 34.1, 29.5]

# Creating a list of predicted values
y_pred = [25.7, 43.0, 35.5, 30.1, 49.8, 54.2, 22.5, 34.2, 38.9, 42.4,
          26.3, 37.6, 46.7, 51.1, 33.5, 27.7, 43.2, 36.9, 33.4, 31.0]