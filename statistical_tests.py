import numpy  as np
import pandas as pd

from scipy import stats


"""
Statistical tests to check whether a distribution is normal.
"""
def apply_dagostino_test(variable, alpha=0.05, return_p=True):

    _, p = stats.normaltest(variable)
    
    result = 'Not normal' if p < alpha else 'Normal'

    return (result, p) if return_p else result


def apply_kurtosis_test(variable, alpha=0.05, return_p=True):

    _, p = stats.kurtosistest(variable)
    
    result = 'Not normal' if p < alpha else 'Normal'

    return (result, p) if return_p else result


def apply_skewness_test(variable, alpha=0.05, return_p=True):

    _, p = stats.skewtest(variable)
    
    result = 'Not normal' if p < alpha else 'Normal'

    return (result, p) if return_p else result


def apply_normal_tests(variable, alpha=0.05, return_p=False, return_kurt_skew=False):

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
    
    dict_result = {'Normal Tests': dict_normal_tests}
    
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
