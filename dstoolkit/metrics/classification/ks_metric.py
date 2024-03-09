from scipy.stats import ks_2samp


def ks_metric(y, col_target='target', col_prob='prob', return_pvalue=False):

    """
    Functions to calculate KS metric

    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

    Parameters
    ----------
    y : DataFrame
        Data frame with target and predicted probability.

    col_target : str
        The name of the columns with the target.

    col_prob : str
        The name of the columns with the predicted probability.

    return_pvalue : bool
        Variable defines whether pvalue will be returned o not.
    
    Returns
    -------
    statistic : float
        KS test statistic.

    pvalue : float
        One-tailed or two-tailed p-value.
    """

    serie_prob_class0 = y.loc[y[col_target] == 0, col_prob]
    serie_prob_class1 = y.loc[y[col_target] == 1, col_prob]

    statistic, pvalue = ks_2samp(serie_prob_class0, serie_prob_class1)

    return (statistic, pvalue) if return_pvalue else statistic

