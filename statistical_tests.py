import numpy as np
import pandas as pd
from scipy import stats


"""
Statistical tests to check whether a distribution is normal.
"""
def dagostino_test(data, alpha=0.05):
    
    _, p = stats.normaltest(data)
    result = 'Not normal' if p < alpha else 'Normal'

    return result


def kurtosis_test(data, alpha=0.05):
    
    _, p = stats.kurtosistest(data)
    result = 'Not normal' if p < alpha else 'Normal'
    
    return f'{result}, {round(p, 4)}'


def skewness_test(data, alpha=0.05):
    
    _, p = stats.skewtest(data)
    result = 'Not normal' if p < alpha else 'Normal'
    
    return f'{result}, {round(p, 4)}'


def normal_tests(data, alpha=0.05):
    
    yj = stats.yeojohnson(data)[0]
    sq = np.sqrt(data)
    cb = np.cbrt(data)
    log = np.log(data)
    
    return pd.DataFrame({
        'Skewness': [stats.skew(data), stats.skew(sq), stats.skew(cb), stats.skew(log), stats.skew(yj)],
        'Skewness Test': [skewness_test(data, alpha=alpha), skewness_test(sq, alpha=alpha), skewness_test(cb, alpha=alpha), skewness_test(log, alpha=alpha), skewness_test(yj, alpha=alpha)],
        'Kurtosis': [stats.kurtosis(data), stats.kurtosis(sq), stats.kurtosis(cb), stats.kurtosis(log), stats.kurtosis(yj)],
        'Kurtosis Test': [kurtosis_test(data, alpha=alpha), kurtosis_test(sq, alpha=alpha), kurtosis_test(cb, alpha=alpha), kurtosis_test(log, alpha=alpha), kurtosis_test(yj, alpha=alpha)],
        'Normal Test': [dagostino_test(data, alpha=alpha), dagostino_test(sq, alpha=alpha), dagostino_test(cb, alpha=alpha), dagostino_test(log, alpha=alpha), dagostino_test(yj, alpha=alpha)]}, 
        index=['default', 'sqrt', 'cuberoot', 'log', 'yeojohnson'])
