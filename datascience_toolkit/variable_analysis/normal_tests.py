import numpy  as np
import pandas as pd

from scipy import stats

from .kurtosis_test  import  apply_kurtosis_test
from .skewness_test  import  apply_skewness_test
from .dagostino_test import apply_dagostino_test


def apply_normal_tests(variable: pd.Series, alpha: float = 0.05, return_p: bool = False, return_kurt_skew: bool = False):

    yj, _ = stats.yeojohnson(variable)
    sq = np.sqrt(variable)
    cb = np.cbrt(variable)
    log = np.log(variable)
    
    dict_normal_tests = {
        'Original': apply_dagostino_test(variable, alpha=alpha, return_p=return_p),
        'Sqrt': apply_dagostino_test(sq, alpha=alpha, return_p=return_p),
        'Cube Root': apply_dagostino_test(cb, alpha=alpha, return_p=return_p),
        'Log': apply_dagostino_test(log, alpha=alpha, return_p=return_p),
        'Yeo Johnson': apply_dagostino_test(yj, alpha=alpha, return_p=return_p)
    }
    
    dict_result = {'Normal Testes': dict_normal_tests}
    
    if return_kurt_skew:
    
        dict_skewness_tests = {
            'Original': apply_skewness_test(variable, alpha=alpha, return_p=return_p),
            'Sqrt': apply_skewness_test(sq, alpha=alpha, return_p=return_p),
            'Cube Root': apply_skewness_test(cb, alpha=alpha, return_p=return_p),
            'Log': apply_skewness_test(log, alpha=alpha, return_p=return_p),
            'Yeo Johnson': apply_skewness_test(yj, alpha=alpha, return_p=return_p)
        }

        dict_kurtosis_tests = {
            'Original': apply_kurtosis_test(variable, alpha=alpha, return_p=return_p),
            'Sqrt': apply_kurtosis_test(sq, alpha=alpha, return_p=return_p),
            'Cube Root': apply_kurtosis_test(cb, alpha=alpha, return_p=return_p),
            'Log': apply_kurtosis_test(log, alpha=alpha, return_p=return_p),
            'Yeo Johnson': apply_kurtosis_test(yj, alpha=alpha, return_p=return_p)
        }
    
        dict_result.update(
            {'Skewness Tests': dict_skewness_tests, 'Kurtosis Tests': dict_kurtosis_tests})

    return dict_result