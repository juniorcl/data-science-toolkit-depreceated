import shap


def get_tree_summary_plot(model, X, selected_features=None):

    if selected_features:

        X = X.loc[:, selected_features]
    
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X)