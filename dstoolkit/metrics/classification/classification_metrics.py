import numpy as np

from .ks_metric import ks_metric
from sklearn.metrics import roc_auc_score


def get_classification_metrics(y, col_target='target', col_prob='prob', decimals=0):

    """
    Function to calculate the regression metrics: R2, RMSE, MAE, MAPE, MedAE

    Parameters
    ----------
    y : DataFrame
        Data frame with target and prediction.

    col_target : str
        The name of the columns with the target.

    col_pred : str
        The name of the columns with the prediction.

    decimals : int
        Number of decimal places to round.
    
    Returns
    -------
    dict_results : Dict
        Dictionary with metric results.
    """

    auc = roc_auc_score(y[col_target], y[col_prob])
    ks = ks_metric(y, col_target=col_target, col_prob=col_prob)

    dict_results = {"ROC AUC": np.round(auc, decimals), "KS": np.round(ks, decimals)}

    return dict_results