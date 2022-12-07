import numpy  as np
import pandas as pd

from scipy import stats


"""
Statistical tests to check whether a distribution is normal.
"""
def dagostino_test(data, alpha=0.05, verbose=0):

    _, p = stats.normaltest(data)
    result = 'Not normal' if p < alpha else 'Normal'

    return result if verbose == 0 else f'{result}, {round(p, 4)}'


def kurtosis_test(data, alpha=0.05, verbose=0):

    _, p = stats.kurtosistest(data)
    result = 'Not normal' if p < alpha else 'Normal'

    return result if verbose == 0 else f'{result}, {round(p, 4)}'


def skewness_test(data, alpha=0.05, verbose=0):

    _, p = stats.skewtest(data)
    result = 'Not normal' if p < alpha else 'Normal'

    return result if verbose == 0 else f'{result}, {round(p, 4)}'


def normal_tests(data, alpha=0.05, verbose=0):

    yj, _ = stats.yeojohnson(data)
    sq = np.sqrt(data)
    cb = np.cbrt(data)
    log = np.log(data)

    result = pd.DataFrame(
        {
            'Skewness': [
                stats.skew(data),
                stats.skew(sq),
                stats.skew(cb),
                stats.skew(log),
                stats.skew(yj)
                ],
            'Skewness Test': [
                skewness_test(data, alpha=alpha, verbose=verbose),
                skewness_test(sq, alpha=alpha, verbose=verbose),
                skewness_test(cb, alpha=alpha, verbose=verbose),
                skewness_test(log, alpha=alpha, verbose=verbose),
                skewness_test(yj, alpha=alpha, verbose=verbose)
                ],
            'Kurtosis': [
                stats.kurtosis(data),
                stats.kurtosis(sq),
                stats.kurtosis(cb),
                stats.kurtosis(log),
                stats.kurtosis(yj)
                ],
            'Kurtosis Test': [
                kurtosis_test(data, alpha=alpha, verbose=verbose),
                kurtosis_test(sq, alpha=alpha, verbose=verbose),
                kurtosis_test(cb, alpha=alpha, verbose=verbose),
                kurtosis_test(log, alpha=alpha, verbose=verbose),
                kurtosis_test(yj, alpha=alpha, verbose=verbose)
                ],
            'Normal Test': [
                dagostino_test(data, alpha=alpha, verbose=verbose),
                dagostino_test(sq, alpha=alpha, verbose=verbose),
                dagostino_test(cb, alpha=alpha, verbose=verbose),
                dagostino_test(log, alpha=alpha, verbose=verbose),
                dagostino_test(yj, alpha=alpha, verbose=verbose)
                ]
         }, index=['default', 'sqrt', 'cuberoot', 'log', 'yeojohnson'])

    return result
