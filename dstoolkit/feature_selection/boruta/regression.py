from BorutaShap import BorutaShap


def boruta_shap_regression(
        X_train, y_train, model=None, target='target', n_trials=100, sample=False, train_or_test='test', normalize=True, verbose=False):
    
    boruta_shap = BorutaShap(model, importance_measure='shap', classification=False)
    boruta_shap.fit(
        X=X_train, y=y_train[target], n_trials=n_trials, sample=sample, 
        train_or_test=train_or_test, normalize=normalize, verbose=verbose)

    list_selected_features = boruta_shap.Subset().columns.tolist()
    
    return list_selected_features