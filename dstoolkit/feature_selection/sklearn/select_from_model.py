from sklearn.feature_selection import SelectFromModel


def select_from_model(estimator, X_train, y_train, target='target', threshold=None, max_features=None):

    sfm = SelectFromModel(estimator, threshold=None, max_features=None)
    sfm.fit(X_train, y_train[target])

    list_selected_features = X_train.loc[:, sfm.get_support()].columns.tolist()

    return list_selected_features