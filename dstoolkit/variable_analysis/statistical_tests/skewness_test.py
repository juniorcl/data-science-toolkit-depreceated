import pandas as pd

from scipy.stats import skewtest


def apply_skewness_test(variable: pd.Series, alpha: float = 0.05, return_p: bool = True):

    _, p = skewtest(variable)
    
    result = 'Not normal' if p < alpha else 'Normal'

    return (result, p) if return_p else result