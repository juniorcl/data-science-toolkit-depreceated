import optuna

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_validate


def tune_params_lightgbm_regression_cv(X_train, y_train, selected_features, n_trials=100, target='target', scoring='r2', direction='maximize'):
    
    def objective(trial):
    
        param = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "bagging_freq": 1,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)
        }
        
        cv_results = cross_validate(
            estimator=LGBMRegressor(**param), 
            scoring=scoring, cv=3,
            X=X_train[selected_features], y=y_train[target])
    
        score_mean = cv_results['test_score'].mean()
    
        return score_mean
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params