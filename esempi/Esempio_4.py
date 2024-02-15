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



def custom_medae(y_true, y_pred):
    
    # Creating an empty list of absolute errors
    absolute_errors = []
    
    # Iterating through actual and predicted values for y
    for true, predicted in zip(y_true, y_pred):
        
        # Computing the differences(i.e., errors)
        error = true - predicted
        # Obtaining the absolute value
        if error < 0:  # If the difference is a negative number,
            error = -error # We obtain the negative of the negative, which is a positive number
        
        absolute_errors.append(error) # Adding absolute value to the list of absolute errors
    
    # Ordering absolute_errors list in ascending order
    sorted_absolute_errors = sorted(absolute_errors)
    # Obtaining the total number of elements in the sorted_absolute_errors list
    n = len(sorted_absolute_errors)
    
    # Obtaining the middle index of the list by dividing the total length of the list by half
    middle = n // 2 # Floor division to return an integer
    
    # We must check if we have an even or odd number of elements
    if n % 2 ==0: # If we have an even number of elements,
        # The median will be equal to the mean of the two elements in the middle of the list
        medae = (sorted_absolute_errors[middle - 1] + sorted_absolute_errors[middle]) / 2
    else:
        # For an odd number of elements, the median will be equal to the value in the middle of the list
        medae = sorted_absolute_errors[middle]    
    return medae


#!/usr/bin/python
#title          :abalone.py
#description    :A testing script for Guozhu Dong's CPXR algorithm on abalone data
#author         :Henry Lin
#version        :0.0.1
#python_version :2.7.6
#================================================================================

#Determing Optimal Threshold


def get_per_class_metrics(ytest, ypred):
    precision_arr = precision_score(ytest, ypred, average=None)
    recall_arr = recall_score(ytest, ypred, average=None)
    f1_arr = f1_score(ytest, ypred, average=None)

    return precision_arr, recall_arr, f1_arr

def get_overall_metrics(ytest, ypred, average="weighted"):
    acc = accuracy_score(ytest, ypred)

    precision = precision_score(ytest, ypred, average=average)
    recall = recall_score(ytest, ypred, average=average)
    f1 = f1_score(ytest, ypred, average=average)

    return acc, precision, recall, f1

def print_scores(acc, precision, recall, f1):
    print("Accuracy: ", acc)
    print("precision score:", precision)
    print("recall score:", recall)
    print("F1 score:", f1)
     

    df=Input("/Users/UR00601/Replatforming/datasets/Z01_Projects/A03_Finalta/Ricavi/A001A01_Base_Mol"),
)
def compute(df):

    # Ricavi da Impieghi : Rateizzazione e Revolving (non riesci a distringuere uno o l'altro)
    impieghi = df.filter(F.col('co_prodotto').isin('7125', '7126', '7603', '7604', '7606', '7607'))\
                 .withColumn('Carte_Di_Credito', F.lit('Rateizzazioni_&_Revolving'))\
                 .withColumn('Mol_da_utilizzare', F.col('mol'))

    # Ricavi da Servizio
    Interchange_fees = ['400016', '400017', '400018', '400019', '700856',
                        '700867', '700891', '760216', '760224', '760228',
                        '760235', '760236', '760238', '760244',  '898748',
                        '898808', '898815', '898823']

    Cash_advance_ATM_fee = ['700794', '700786', '898743', '898805', '898820']

    Membership_fees = ['700760', '700763', '700806', '700807', '700860', '700862', '760212', '760221',
                       '760232', '760240', '898740', '898813', '898828', '898835']

    Currency_Exchange_Fee = ['700874', '700789', '700872', '700793', '700873',
                             '898811', '898818', '898826', '898830', '898832']

    df = df.withColumn('Carte_Di_Credito', F.when(F.col('tp_oper').isin(Interchange_fees), F.lit('Interchange_fees'))\
                                            .when(F.col('tp_oper').isin(Cash_advance_ATM_fee), F.lit('Cash_advance_ATM_fee'))\
                                            .when(F.col('tp_oper').isin(Membership_fees), F.lit('Membership_fees'))\
                                            .when(F.col('tp_oper').isin(Currency_Exchange_Fee), F.lit('Currency_Exchange_Fee'))
                      )\
           .withColumn('Mol_da_utilizzare', F.when(F.col('tp_oper').isin(Interchange_fees), F.col('cm_rp_att'))\
                                             .when(F.col('tp_oper').isin(Cash_advance_ATM_fee), F.col('cm_rp_att'))\
                                             .when(F.col('tp_oper').isin(Membership_fees), F.col('cm_rp_att'))\
                                             .when(F.col('tp_oper').isin(Currency_Exchange_Fee), F.col('cm_rp_att'))
                      )

    df = df.unionByName(impieghi)

    return df



    # calcolare i ventili
    w = Window.orderBy(df.score)
    df = df.select('*', ceil(5 * percent_rank().over(w)).alias("ventile"))
    
    

    canone = B01B07_Canone.withColumn('card_contr_id', F.lpad(F.col('KCCEM_10010_E_CRE_RAPPORTO'), 14, '0'))\
                          .select('card_contr_id',
                                  'KCCEM_10090_E_CRE_IM_IMPMOV',
                                  'KCCEM_10140_E_CRE_RIF_MOV', 
                                  'KCCEM_10150_E_CRE_DES_MOV',
                                  'KCCEM_10120_E_CRE_DT_REG')

    df = B01B01_Stock_Primario.join(canone, on='card_contr_id')
    return df



    flussi=Input("/uci/cido_consumer/Volumes & Sales/Data/cbk_volume_sales_movements_tot_asset_globe")
)
def compute(flussi):

    dt_rif = flussi.filter(F.col('snapshot_date') <= '2023-12-29').agg(F.max(F.col('snapshot_date'))).collect()[0][0]  # noqa

    df = (
        flussi
        .filter(F.col("snapshot_date") == dt_rif)
        .filter("mkt_prod_hier_lev02_cd in  ('10')")
        .filter(F.col("macro_area").isin(['RETAIL', 'BUDDY',  'PRIVATE']))
#        .filter(F.col("shadow_in") == 'N')

    )

    entrate = ['MOVA', 'MOVB', 'MOVS', 'MOVW', 'MOVG']
    uscite = ['MOVD', 'MOVC', 'MOVV', 'MOVX', 'MOVH']

    df = df.filter("mkt_prod_hier_lev02_cd in  ('10')")\
           .withColumn('tipo_movimento', F.when(F.col('mis_data_tp').isin(entrate), F.lit("entrate"))\
                                          .when(F.col('mis_data_tp').isin(uscite), F.lit("uscite"))
                                          .otherwise(F.lit('Non_Classificato')))
 
    df = (
        df
        .withColumn('stock_raccolta_diretta', F.expr("case when mkt_prod_hier_lev02_cd ='10' then curr_year_prog_mov_vl else 0 end"))
        .withColumn('stock_raccolta_diretta_vista', F.expr("case when mkt_prod_hier_lev03_cd='100'\
            then curr_year_prog_mov_vl else 0 end"))
        .withColumn('stock_raccolta_diretta_a_tempo', F.expr("case when mkt_prod_hier_lev03_cd ='102' then\
            curr_year_prog_mov_vl else 0 end"))
        .withColumn('stock_raccolta_diretta_obbigazioni_proprie', F.expr("case when mkt_prod_hier_lev03_cd='104'\
            then curr_year_prog_mov_vl else 0 end"))
    )
    return df



df=Input("/uci/cido_consumer/Commercial_Sales/data/Commercial_Sales_Mch_2023"),
)
def compute(df):
    df = df.filter((F.col('prod_lev_02') == 'BANCASSURANCE') |
                   (F.col('prod_lev_01') == 'PROTEZIONE'))
    return df


    