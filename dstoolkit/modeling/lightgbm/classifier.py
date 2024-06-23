import numpy as np

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate

from ...optimization.optuna.lightgbm                                import tune_params_lgbm_classifier_cv
from ...variable_analysis.shap.summary_shap                         import get_tree_summary_plot
from ...feature_selection.boruta.classification                     import boruta_shap_classification
from ...feature_selection.sklearn.select_from_model                 import select_from_model
from ...metrics.classification.classification_metrics               import get_classification_metrics
from ...variable_analysis.feature_importance.feature_importance     import get_tree_feature_importance
from ...variable_analysis.feature_importance.permutation_importance import get_permutation_importance


def fit_lgbm_classifier_cv(X_train, y_train, X_test, y_test, target, cv=3, params=None, random_state=42):

    cat_columns = X_train.select_dtypes(include='object')

    if any(cat_columns):

        print('---------------> Feature Engineering')
        
        X_train[cat_columns] = X_train[cat_columns].astype("category")
        X_test[cat_columns] = X_test[cat_columns].astype("category")

    print('---------------> Modeling')

    init_params = {'verbosity': -1, 'random_state': random_state, 'n_jobs': -1}
    
    if params:
        
        init_params.update(params)

    model = LGBMClassifier(**init_params)

    list_class_metrics = ['roc_auc']

    cv_results = cross_validate(estimator=model, X=X_train, y=y_train[target], cv=cv, scoring=list_class_metrics)

    auc_mean = np.round(cv_results['test_roc_auc'].mean(), 2)

    model.fit(X_train, y_train[target])
    y_test['prob'] = model.predict_proba(X_test)[:, 1]

    dict_results = get_classification_metrics(y_test, target, decimals=2)
    
    auc, ks = dict_results['ROC AUC'], dict_results['KS']

    print('---------------> Metrics')

    print(f"Cross Validation  ROC AUC: {auc_mean}, KS: -")
    print(f"Test  Validation  ROC AUC: {auc}, KS: {ks}")

    return model


def automl_lgbm_classifier_cv(X_train, y_train, X_test, y_test, selection_method='sfm', target='target', cv=3, n_trials=100, scoring='roc_auc', direction='maximize', random_state=42):

    dict_results = {}
    
    print('--------> Standard Model')

    standard_model = fit_lgbm_classifier_cv(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, target=target, cv=cv, random_state=random_state)

    dict_results['standard_model'] = standard_model

    print('\n--------> Feature Selection')

    if selection_method == 'sfm':

        print('---------------> Select From Model')
        
        list_selected_features = select_from_model(
            estimator=LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=-1), X_train=X_train, y_train=y_train, target=target)

        dict_results['selected_features'] = list_selected_features
    
    elif selection_method == 'boruta':

        print('---------------> Boruta Shap')

        list_selected_features = boruta_shap_classification(
            X_train=X_train, y_train=y_train, model=LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=-1), 
            n_trials=100, sample=False, train_or_test='test', normalize=True, verbose=False, target=target)

        dict_results['selected_features'] = list_selected_features
    
    selected_features_model = fit_lgbm_classifier_cv(
        X_train=X_train[list_selected_features], y_train=y_train, 
        X_test=X_test[list_selected_features], y_test=y_test, target=target, cv=cv, random_state=random_state)
    
    dict_results['selected_features_model'] = selected_features_model
    
    print('\n--------> Hyperparameter Tuning')
    
    params = tune_params_lgbm_classifier_cv(
        X_train, y_train, list_selected_features, n_trials=n_trials, 
        target=target, scoring=scoring, direction=direction, random_state=random_state)

    dict_results['best_params'] = params
    
    model = fit_lgbm_classifier_cv(
        X_train=X_train[list_selected_features], y_train=y_train, 
        X_test=X_test[list_selected_features], y_test=y_test, target=target, cv=cv, params=params, random_state=random_state)

    dict_results['model'] = model
    
    print('\n--------> Feature Importance')
    
    df_imp = get_tree_feature_importance(model, X_train[list_selected_features])

    dict_results['feature_importance'] = df_imp

    print('\n--------> Permutation Importance')

    df_perm = get_permutation_importance(model, X_test[list_selected_features], y_test[target], scoring=scoring, random_state=random_state, n_repeats=5)

    dict_results['permutation_importance'] = df_perm
    
    print('\n--------> Shap Values')
    
    get_tree_summary_plot(model, X_train[list_selected_features])

    return dict_results