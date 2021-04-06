import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest, skewtest, kurtosistest, shapiro


def normality_test(x):
    '''function with multiple tests to define 
    if a distribution is normal. The distribution 
    goes out of its natural form and is subjected 
    to calculations to acquire a normal form.'''

    yeo_johnson = PowerTransformer(standardize=False)
    box_cox = PowerTransformer(method='box-cox', standardize=False)
    
    sq = np.sqrt(x)
    cb = np.cbrt(x)
    log = np.log(x)
    yj = yeo_johnson.fit_transform(x)
    bx = box_cox.fit_transform(x)

    tests = ["D'Agostino p-value", "Shapiro-Wilk p-value", "Kurtosis p-value", "Skew p-value"]
    transformations = [x, sq, cb, log, yj, bx]
    functions = [normaltest, shapiro, kurtosistest, skewtest]
    
    df_dict = {}
    for test, func in zip(tests, functions):
        df_dict[test] = []
        
        for data in transformations:
            if func != shapiro:
                 p = func(data).pvalue[0]
            else:
                p = func(data).pvalue
            
            df_dict[test].append(p)
    
    
    return pd.DataFrame(df_dict, index=['default', 'sqrt', 'cuberoot', 'log', 'yeo-johnson', 'box-cox'])