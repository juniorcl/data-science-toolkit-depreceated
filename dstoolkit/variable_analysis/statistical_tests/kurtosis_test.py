import pandas as pd

from scipy.stats import kurtosistest


def apply_kurtosis_test(variable: pd.Series, alpha: float = 0.05, return_p: bool = True):

    _, p = kurtosistest(variable)
    
    result = 'Not normal' if p < alpha else 'Normal'

    return (result, p) if return_p else result