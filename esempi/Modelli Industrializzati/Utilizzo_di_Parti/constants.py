from sklearn.metrics import auc, precision_recall_curve


def AUPRC(y_true, y_pred, pos_label=1):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label)
    return auc(recall, precision)


def AUPRC_lgbm(y_true, y_pred):
    '''
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    Custom eval function expects a callable with following signatures:
        - func(y_true, y_pred)
        - func(y_true, y_pred, weight)
        - func(y_true, y_pred, weight, group)
    returns:
        - (eval_name: str, eval_result: float, is_higher_better: bool)
        - list of (eval_name: str, eval_result: float, is_higher_better: bool)
    '''
    return 'AUPRC', AUPRC(y_true, y_pred), True