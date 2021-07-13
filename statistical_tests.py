"""
A set of functions with estatistical tests.
"""
from scipy import stats

def dagostini_test(x):

    alpha = 0.05
    _, p = stats.normaltest(x)
    result = "Normal" if p > alpha else "Not normal"

    return result, round(p, 4)

