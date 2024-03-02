import pandas as pd

from scipy.stats import normaltest


def apply_dagostino_test(variable: pd.Series, alpha: float = 0.05, return_p: bool = True):

    _, p = normaltest(variable)
    
    result = 'Not normal' if p < alpha else 'Normal'

    return (result, p) if return_p else result