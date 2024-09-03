import numpy as np

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate

from ...optimization.optuna.lightgbm                                import tune_params_lgbm_classifier_cv, tune_params_lgbm_classifier
from ...variable_analysis.shap.summary_shap                         import get_tree_summary_plot
from ...feature_selection.boruta.classification                     import boruta_shap_classification
from ...feature_selection.sklearn.select_from_model                 import select_from_model
from ...metrics.classification.classification_metrics               import get_classification_metrics
from ...variable_analysis.feature_importance.feature_importance     import get_tree_feature_importance
from ...variable_analysis.feature_importance.permutation_importance import get_permutation_importance


def fit_lgbm_classifier_cv(X_train, y_train, X_test, y_test, target, selected_features=None, cv=3, params=None, random_state=42):

    if selected_features:

        X_train = X_train.loc[:, selected_features]
        X_test = X_test.loc[:, selected_features]

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

    auc_mean = np.round(cv_results['test_roc_auc'].mean(), 4)

    model.fit(X_train, y_train[target])
    y_test['prob'] = model.predict_proba(X_test)[:, 1]

    dict_results = get_classification_metrics(y_test, target, decimals=4)
    
    auc, ks = dict_results['ROC AUC'], dict_results['KS']

    print('---------------> Metrics')

    print(f"Cross Validation  ROC AUC: {auc_mean}, KS: -")
    print(f"Test  Validation  ROC AUC: {auc}, KS: {ks}")

    return model


def automl_lgbm_classifier_cv(
    X_train, y_train, X_test, y_test, selection_method='sfm', target='target', cv=3, n_trials=100, scoring='roc_auc', direction='maximize', random_state=42):

    dict_results = {}
    
    print('--------> Standard Model')

    standard_model = fit_lgbm_classifier_cv(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_features=None, target=target, cv=cv, random_state=random_state)

    dict_results['standard_model'] = standard_model

    if selection_method == 'sfm':

        print('\n--------> Feature Selection', '\n---------------> Select From Model')
        
        list_selected_features = select_from_model(
            estimator=LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=-1), X=X_train, y=y_train, target=target)

        dict_results['selected_features'] = list_selected_features

        selected_features_model = fit_lgbm_classifier_cv(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            selected_features=list_selected_features, target=target, cv=cv, random_state=random_state)
        
        dict_results['selected_features_model'] = selected_features_model
    
    elif selection_method == 'boruta':

        print('\n--------> Feature Selection', '\n---------------> Boruta Shap')

        list_selected_features = boruta_shap_classification(
            X_train=X_train, y_train=y_train, model=LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=-1), 
            n_trials=100, sample=False, train_or_test='test', normalize=True, verbose=False, target=target)

        dict_results['selected_features'] = list_selected_features

        selected_features_model = fit_lgbm_classifier_cv(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            selected_features=list_selected_features, target=target, cv=cv, random_state=random_state)

        dict_results['selected_features_model'] = selected_features_model

    else:

        list_selected_features = None
    
    print('\n--------> Hyperparameter Tuning')
    
    params = tune_params_lgbm_classifier_cv(
        X_train, y_train, selected_features=list_selected_features, n_trials=n_trials, target=target, scoring=scoring, direction=direction, random_state=random_state)

    dict_results['best_params'] = params
    
    model = fit_lgbm_classifier_cv(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, params=params,
        selected_features=list_selected_features, target=target, cv=cv, random_state=random_state)

    dict_results['model'] = model
    
    print('\n--------> Feature Importance')
    
    df_imp = get_tree_feature_importance(model, X_train, selected_features=list_selected_features)

    dict_results['feature_importance'] = df_imp

    print('\n--------> Permutation Importance')

    df_perm = get_permutation_importance(model, X_train, y_train, target=target, selected_features=list_selected_features, scoring=scoring, random_state=random_state, n_repeats=5)

    dict_results['permutation_importance'] = df_perm
    
    print('\n--------> Shap Values')
    
    get_tree_summary_plot(model, X_train, selected_features=list_selected_features)

    return dict_results


def fit_lgbm_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, target='target', selected_features=None, params=None, random_state=42):

    if selected_features:

        X_train = X_train.loc[:, selected_features]
        X_valid = X_valid.loc[:, selected_features]
        X_test = X_test.loc[:, selected_features]

    cat_columns = X_train.select_dtypes(include='object')

    if any(cat_columns):

        print('---------------> Feature Engineering')
        
        X_train[cat_columns] = X_train[cat_columns].astype("category")
        X_valid[cat_columns] = X_valid[cat_columns].astype("category")
        X_test[cat_columns] = X_test[cat_columns].astype("category")

    print('---------------> Modeling')

    init_params = {'verbosity': -1, 'random_state': random_state, 'n_jobs': -1}
    
    if params:
        
        init_params.update(params)

    model = LGBMClassifier(**init_params)

    model.fit(X_train, y_train[target])

    y_train['prob'] = model.predict_proba(X_train)[:, 1]
    
    y_valid['prob'] = model.predict_proba(X_valid)[:, 1]
    
    y_test['prob'] = model.predict_proba(X_test)[:, 1]

    dict_train_results = get_classification_metrics(y_train, target, decimals=2)
    dict_valid_results = get_classification_metrics(y_valid, target, decimals=2)
    dict_test_results = get_classification_metrics(y_test, target, decimals=2)
    

    print('---------------> Metrics')

    print(f"Training    ROC AUC: {dict_train_results['ROC AUC']}, KS: {dict_train_results['KS']}")
    print(f"Validation  ROC AUC: {dict_valid_results['ROC AUC']}, KS: {dict_valid_results['KS']}")
    print(f"Testing     ROC AUC: {dict_test_results['ROC AUC']}, KS: {dict_test_results['KS']}")
    
    return model


def automl_lgbm_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, selection_method='sfm', target='target', n_trials=100, direction='maximize', random_state=42):

    dict_results = {}
    
    print('--------> Standard Model')

    standard_model = fit_lgbm_classifier(
        X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, selected_features=None, target=target, random_state=random_state)

    dict_results['standard_model'] = standard_model

    if selection_method == 'sfm':

        print('\n--------> Feature Selection', '\n---------------> Select From Model')
        
        list_selected_features = select_from_model(estimator=LGBMClassifier(verbosity=-1, random_state=42, n_jobs=-1), X=X_train, y=y_train, target=target)

        dict_results['selected_features'] = list_selected_features

        selected_features_model = fit_lgbm_classifier(
            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, selected_features=list_selected_features, target=target, random_state=random_state)
    
        dict_results['selected_features_model'] = selected_features_model
    
    elif selection_method == 'boruta':

        print('\n--------> Feature Selection', '\n---------------> Boruta Shap')

        list_selected_features = boruta_shap_classification(
            X=X_train, y=y_train, model=LGBMClassifier(verbosity=-1, random_state=42, n_jobs=-1), 
            n_trials=100, sample=False, train_or_test='test', normalize=True, verbose=False)

        dict_results['selected_features'] = list_selected_features

        selected_features_model = fit_lgbm_classifier(
            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, 
            selected_features=list_selected_features, target=target, random_state=random_state)
    
        dict_results['selected_features_model'] = selected_features_model
    
    else:

        list_selected_features = None

    print('\n--------> Hyperparameter Tuning')
    
    params = tune_params_lgbm_classifier(
        X_train, y_train, X_valid, y_valid, selected_features=list_selected_features, n_trials=n_trials, 
        target=target, direction=direction, random_state=random_state)

    dict_results['best_params'] = params
    
    model = fit_lgbm_classifier(
        X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, 
        selected_features=list_selected_features, target=target, random_state=random_state, params=params)

    dict_results['model'] = model
    
    print('\n--------> Feature Importance')
    
    df_imp = get_tree_feature_importance(model, X_train, selected_features=list_selected_features)

    dict_results['feature_importance'] = df_imp

    print('\n--------> Permutation Importance')

    df_perm = get_permutation_importance(model, X_test, y_test, target=target, selected_features=list_selected_features, random_state=random_state, n_repeats=5)

    dict_results['permutation_importance'] = df_perm
    
    print('\n--------> Shap Values')
    
    get_tree_summary_plot(model, X_train, selected_features=list_selected_features)

    return dict_results