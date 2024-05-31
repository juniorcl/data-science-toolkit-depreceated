import numpy as np

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_validate

from ...variable_analysis.shap.summary_shap                     import get_tree_shap_values
from ...metrics.regression.regression_metrics                   import get_regression_metrics
from ...optimization.optuna.hyperparameter_tuning               import tune_params_lightgbm_regression_cv
from ...feature_selection.boruta.boruta_regression              import boruta_shap_regression
from ...feature_selection.sklearn.select_from_model             import select_from_model
from ...variable_analysis.feature_importance.feature_importance import get_tree_feature_importance


def fit_lightgbm_regressor_cv(X_train, y_train, X_test, y_test, target, cv=3, params=None, random_state=42):

    cat_columns = X_train.select_dtypes(include='object')

    if any(cat_columns):

        print('---------------> Feature Engineering')
        
        X_train[cat_columns] = X_train[cat_columns].astype("category")
        X_test[cat_columns] = X_test[cat_columns].astype("category")

    print('---------------> Modeling')

    init_params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'random_state': random_state, "bagging_freq": 1, 'n_jobs': -1}
    
    if params:
        
        params.update(**init_params)
        model = LGBMRegressor(**params)
    
    else:

        model = LGBMRegressor(**init_params)
    
    cv_results = cross_validate(
        estimator=model, X=X_train, y=y_train['target'], cv=3,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'neg_mean_absolute_percentage_error'])

    r2_mean = np.round(cv_results['test_r2'].mean(), 2)
    mae_mean = np.round(cv_results['test_neg_mean_absolute_error'].mean() * -1, 2)
    rmse_mean = np.round(cv_results['test_neg_root_mean_squared_error'].mean() * -1, 2)
    medae_mean = np.round(cv_results['test_neg_median_absolute_error'].mean() * -1, 2)
    mape_mean = np.round(cv_results['test_neg_mean_absolute_percentage_error'].mean() * -1, 2)

    model.fit(X_train, y_train[target])
    y_test['pred'] = model.predict(X_test)

    dict_results = get_regression_metrics(y_test, target, decimals=2)
    
    r2, mae, rmse, mape, medae = dict_results['R2'], dict_results['MAE'], dict_results['RMSE'], dict_results['MAPE'], dict_results['MedAE']

    print('---------------> Metrics')

    print(f"Cross Validation  R2: {r2_mean}, MAE: {mae_mean}, RMSE: {rmse_mean}, MAPE: {mape_mean}, MedAE: {medae_mean}")
    print(f"Test  Validation  R2: {r2}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, MedAE: {medae}")


def auto_modeling_lightgbm_regressor_cv(
    X_train, y_train, X_test, y_test, selection_method='sfm', target='target', cv=3, n_trials=100, scoring='r2', direction='maximize'):

    print('--------> Standard Model')

    fit_lightgbm_regressor_cv(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, target=target, cv=cv)

    print()
    print('--------> Feature Selection')

    if selection_method == 'sfm':

        print('---------------> Select From Model')
        
        list_selected_features = select_from_model(
            estimator=LGBMRegressor(verbosity=-1, random_state=42, n_jobs=-1), X_train=X_train, y_train=y_train, target=target)

    elif selection_method == 'boruta':

        print('---------------> Boruta Shap')

        list_selected_features = boruta_shap_regression(
            model=LGBMRegressor(verbosity=-1, random_state=42, n_jobs=-1), X_train=X_train, n_trials=100, 
            sample=False, train_or_test='test', normalize=True, verbose=False)

    else:

        list_selected_features = X_train.columns.tolist()
    
    fit_lightgbm_regressor_cv(
        X_train=X_train[list_selected_features], y_train=y_train, 
        X_test=X_test[list_selected_features], y_test=y_test, target=target, cv=cv)

    print()
    print('--------> Hyperparameter Tuning')
    params = tune_params_lightgbm_regression_cv(
        X_train, y_train, list_selected_features, n_trials=n_trials, target='target', scoring=scoring, direction=direction)
    
    fit_lightgbm_regressor_cv(
        X_train=X_train[list_selected_features], y_train=y_train, 
        X_test=X_test[list_selected_features], y_test=y_test, target='target', cv=cv, params=params)

    print()
    print('--------> Final Modeling')
    model = LGBMRegressor(**params)
    model.fit(X_train[list_selected_features], y_train[target])

    print()
    print('--------> Feature Importance')
    df_imp = get_tree_feature_importance(model, X_train[list_selected_features])

    print()
    print('--------> Shap Values')
    get_tree_shap_values(model, X_train[list_selected_features])
    
    return model, df_imp