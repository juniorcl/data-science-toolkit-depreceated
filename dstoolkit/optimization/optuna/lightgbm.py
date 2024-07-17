import optuna
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate


def tune_params_lgbm_regressor_cv(X_train, y_train, selected_features=None, n_trials=100, target='target', scoring='r2', direction='maximize', random_state=42):

    if selected_features:

        X_train = X_train.loc[:, selected_features]
    
    def objective(trial):
    
        param = {
            "objective": "regression",
            "metric": scoring,
            "verbosity": -1,
            "bagging_freq": 1,
            "n_jobs": -1,
            "random_state": random_state,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)
        }
        
        cv_results = cross_validate(
            estimator=LGBMRegressor(**param), scoring=scoring, cv=3, X=X_train, y=y_train[target])
    
        score_mean = cv_results['test_score'].mean()
    
        return score_mean
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = {'objective': 'regression', 'verbosity': -1, 'random_state': random_state, 'metric': scoring, 'n_jobs': -1}
    best_params.update(study.best_params)
    
    return best_params


def tune_params_lgbm_classifier_cv(X_train, y_train, selected_features=None, n_trials=100, target='target', scoring='roc_auc', direction='maximize', random_state=42):

    def objective(trial):
    
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": random_state,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
        }
        
        cv_results = cross_validate(
            estimator=LGBMClassifier(**param), scoring=scoring, cv=3,
            X=X_train[selected_features], y=y_train[target])
    
        score_mean = cv_results['test_score'].mean()
    
        return score_mean
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "n_jobs": -1, "random_state": random_state}
    best_params.update(study.best_params)
    
    return best_params


def tune_params_lgbm_regressor(X_train, y_train, X_valid, y_valid, selected_features=None, n_trials=100, target='target', scoring='r2', direction='maximize', random_state=42):

    if selected_features:

        X_train = X_train.loc[:, selected_features]
        X_valid = X_valid.loc[:, selected_features]
    
    def objective(trial):
    
        param = {
            "objective": "regression",
            "metric": scoring,
            "verbosity": -1,
            "bagging_freq": 1,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)
        }
        
        model = LGBMRegressor(**param)
        model.fit(X_train, y_train[target])

        y_valid['pred'] = model.predict(X_valid)

        match scoring:
            
            case 'r2':
                score = r2_score(y_valid[target], y_valid['pred'])
            
            case 'mae':
                score = mean_absolute_error(y_valid[target], y_valid['pred'])

            case 'rmse':
                score = mean_squared_error(y_valid[target], y_valid['pred'])
                score = np.sqrt(score)

            case _:
                score = r2_score(y_valid[target], y_valid['pred'])
    
        return score
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = {'objective': 'regression', 'verbosity': -1, 'random_state': random_state, 'metric': scoring, 'n_jobs': -1}
    best_params.update(study.best_params)
    
    return best_params