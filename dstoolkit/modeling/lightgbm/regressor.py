import numpy as np

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_validate

from ...optimization.optuna.lightgbm                                import tune_params_lgbm_regressor_cv
from ...feature_selection.boruta.regression                         import boruta_shap_regression
from ...variable_analysis.shap.summary_shap                         import get_tree_summary_plot
from ...metrics.regression.regression_metrics                       import get_regression_metrics
from ...feature_selection.sklearn.select_from_model                 import select_from_model
from ...variable_analysis.feature_importance.feature_importance     import get_tree_feature_importance
from ...variable_analysis.feature_importance.permutation_importance import get_permutation_importance


def fit_lgbm_regressor_cv(X_train, y_train, X_test, y_test, target, selected_features=None, cv=3, params=None, random_state=42):

    if selected_features:

        X_train = X_train.loc[:, selected_features]
        X_test = X_test.loc[:, selected_features]

    cat_columns = X_train.select_dtypes(include='object')

    if any(cat_columns):

        print('---------------> Feature Engineering')
        
        X_train[cat_columns] = X_train[cat_columns].astype("category")
        X_test[cat_columns] = X_test[cat_columns].astype("category")

    print('---------------> Modeling')

    init_params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'random_state': random_state, "bagging_freq": 1, 'n_jobs': -1}
    
    if params:
        
        init_params.update(params)

    model = LGBMRegressor(**init_params)
    
    list_reg_scores = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'neg_mean_absolute_percentage_error']

    cv_results = cross_validate(estimator=model, X=X_train, y=y_train[target], cv=cv, scoring=list_reg_scores)

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

    return model


def automl_lgbm_regressor_cv(X_train, y_train, X_test, y_test, selection_method='sfm', target='target', cv=3, n_trials=100, scoring='r2', direction='maximize', random_state=42):

    dict_results = {}
    
    print('--------> Standard Model')

    standard_model = fit_lgbm_regressor_cv(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_features=None, target=target, cv=cv, random_state=random_state)

    dict_results['standard_model'] = standard_model

    if selection_method == 'sfm':

        print('\n--------> Feature Selection', '\n---------------> Select From Model')
        
        list_selected_features = select_from_model(
            estimator=LGBMRegressor(verbosity=-1, random_state=random_state, n_jobs=-1), X_train=X_train, y_train=y_train, target=target)

        dict_results['selected_features'] = list_selected_features

        selected_features_model = fit_lgbm_regressor_cv(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            selected_features=list_selected_features, target=target, cv=cv, random_state=random_state)
    
        dict_results['selected_features_model'] = selected_features_model
    
    elif selection_method == 'boruta':

        print('\n--------> Feature Selection', '\n---------------> Boruta Shap')

        list_selected_features = boruta_shap_regression(
            X_train=X_train, y_train=y_train, model=LGBMRegressor(verbosity=-1, random_state=random_state, n_jobs=-1), 
            n_trials=100, sample=False, train_or_test='test', normalize=True, verbose=False)

        dict_results['selected_features'] = list_selected_features

        selected_features_model = fit_lgbm_regressor_cv(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            selected_features=list_selected_features, target=target, cv=cv, random_state=random_state)
    
        dict_results['selected_features_model'] = selected_features_model
    
    else:

        list_selected_features = None

    print('\n--------> Hyperparameter Tuning')
    
    params = tune_params_lgbm_regressor_cv(
        X_train, y_train, selected_features=list_selected_features, n_trials=n_trials, 
        target=target, scoring=scoring, direction=direction, random_state=random_state)

    dict_results['best_params'] = params
    
    model = fit_lgbm_regressor_cv(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
        selected_features=list_selected_features, target=target, cv=cv, params=params)

    dict_results['model'] = model
    
    print('\n--------> Feature Importance')
    
    df_imp = get_tree_feature_importance(model, X_train, selected_features=list_selected_features)

    dict_results['feature_importance'] = df_imp

    print('\n--------> Permutation Importance')

    df_perm = get_permutation_importance(
        model, X_test, y_test, target=target, selected_features=list_selected_features, scoring=scoring, random_state=random_state, n_repeats=5)

    dict_results['permutation_importance'] = df_perm
    
    print('\n--------> Shap Values')
    
    get_tree_summary_plot(model, X_train, selected_features=list_selected_features)

    return dict_results