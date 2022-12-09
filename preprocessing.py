import numpy  as np
import pandas as pd
from typing import List
from scipy import stats


class GaussianTransformer(object):

    """
    Class that tries to transform a set of variables into a Gaussian distribution 
    using: log, sqrt, cbrt and Yeo-Johnson.

    If theres no variable that has been trasformed into a Gaussian distribution
    no new variable will be added.

    Parameters
    ----------
    X: Pandas Data Frame
    variables: List of variables
    pvalue: Float that is the threshold to define a distribution as normal
    drop_original: Bool to define if the original variable will be removed
    keep_max: Bool to define if only the best alpha will remain

    Return
    ----------
    X_new: New Pandas Data Frame
    
    Author
    ----------
    Created by Clébio Júnior (github.com/juniorcl)
    """

    def __init__(self, variables: List, pvalue: float = 0.05, drop_original: bool = False, keep_max: bool = False):

        self.pvalue_ = pvalue
        self.variables_ = variables
        self.drop_original_ = drop_original
        self.keep_max_ = keep_max
        self.alpha_results_ = {}
        self.lmbdas_ = {}

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.loc[:, self.variables_]

        for variable in self.variables_:

            self.alpha_results_[variable] = {}
            self.lmbdas_[variable] = {}

            _, alpha = stats.normaltest(X[variable])
            self.alpha_results_[variable]['original'] = alpha
            
            transformed_data, lmbda = stats.yeojohnson(X[variable])
            _, alpha = stats.normaltest(transformed_data)
            self.alpha_results_[variable]['yj'] = alpha
            self.lmbdas_[variable] = lmbda

            transformed_data = np.log(X[variable])
            _, alpha = stats.normaltest(transformed_data)
            self.alpha_results_[variable]['log'] = alpha

            transformed_data = np.sqrt(X[variable])
            _, alpha = stats.normaltest(transformed_data)
            self.alpha_results_[variable]['sqrt'] = alpha

            transformed_data = np.cbrt(X[variable])
            _, alpha = stats.normaltest(transformed_data)
            self.alpha_results_[variable]['cbrt'] = alpha

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.loc[:, self.variables_]

        for variable in self.variables_:
            
            dict_alphas = self.alpha_results_[variable]
            
            is_original_normal = dict_alphas['original'] > self.pvalue_
            is_any_true = any(list(map(lambda i: i > self.pvalue_, dict_alphas.values())))

            if is_any_true and is_original_normal == False:
                
                if dict_alphas['yj'] > self.pvalue_:
                    
                    X[variable + '_yj'] = stats.yeojohnson(X[variable], lmbda=self.lmbdas_[variable]) 

                if dict_alphas['log'] > self.pvalue_:
                    
                    X[variable + '_log'] = np.log(X[variable]) 

                if dict_alphas['sqrt'] > self.pvalue_:
                    
                    X[variable + '_sqrt'] = np.sqrt(X[variable]) 

                if dict_alphas['cbrt'] > self.pvalue_:
                    
                    X[variable + '_cbrt'] = np.cbrt(X[variable])

                if self.keep_max_:

                    filt = {f'{variable}_{f}': v for f, v in dict_alphas.items() if f != 'original' and v > self.pvalue_}
                    
                    columns_to_drop = sorted(filt, key=filt.get)[:-1]
                    X.drop(columns=columns_to_drop, inplace=True)

                if self.drop_original_:
                    
                    X.drop(columns=variable, inplace=True)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:

        self.fit(X)
        return self.transform(X)