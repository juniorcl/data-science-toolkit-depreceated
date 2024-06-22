from sklearn.feature_selection import SelectFromModel


def select_from_model(estimator, X, y, target='target', threshold=None, max_features=None):

    sfm = SelectFromModel(estimator, threshold=threshold, max_features=max_features)
    sfm.fit(X, y[target])

    list_selected_features = X.loc[:, sfm.get_support()].columns.tolist()

    return list_selected_features