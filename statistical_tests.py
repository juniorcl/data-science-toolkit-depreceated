import numpy as np
import pandas as pd
from scipy import stats


"""
Statistical tests to check whether a distribution is normal.
"""
def dagostino_test(data):
    
    alpha = 0.05
    _, p = stats.normaltest(data)
    result = 'Not normal' if p < alpha else 'Normal'

    return result


def kurtosis_test(data):
    
    alpha = 0.05
    _, p = stats.kurtosistest(data)
    result = 'Not normal' if p < alpha else 'Normal'
    
    return f'{result}, {round(p, 4)}'


def skewness_test(data):
    
    alpha = 0.05
    _, p = stats.skewtest(data)
    result = 'Not normal' if p < alpha else 'Normal'
    
    return f'{result}, {round(p, 4)}'


def normal_tests(data):
    
    yj = stats.yeojohnson(data)[0]
    sq = np.sqrt(data)
    cb = np.cbrt(data)
    log = np.log(data)
    
    return pd.DataFrame({
        'Skewness': [stats.skew(data), stats.skew(sq), stats.skew(cb), stats.skew(log), stats.skew(yj)],
        'Skewness Test': [skewness_test(data), skewness_test(sq), skewness_test(cb), skewness_test(log), skewness_test(yj)],
        'Kurtosis': [stats.kurtosis(data), stats.kurtosis(sq), stats.kurtosis(cb), stats.kurtosis(log), stats.kurtosis(yj)],
        'Kurtosis Test': [kurtosis_test(data), kurtosis_test(sq), kurtosis_test(cb), kurtosis_test(log), kurtosis_test(yj)],
        'Normal Test': [dagostino_test(data), dagostino_test(sq), dagostino_test(cb), dagostino_test(log), dagostino_test(yj)]}, 
        index=['default', 'sqrt', 'cuberoot', 'log', 'yeojohnson'])

