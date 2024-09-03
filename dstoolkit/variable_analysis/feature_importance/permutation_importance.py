import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def get_permutation_importance(model, X, y, scoring, target='target', selected_features=None, random_state=42, n_repeats=5):

    if selected_features:

        X = X.loc[:, selected_features]

    dict_results = permutation_importance(
        estimator=model, X=X, y=y[target], scoring=scoring, random_state=random_state, n_repeats=n_repeats, n_jobs=-1)

    sorted_importances_idx = dict_results.importances_mean.argsort()
    
    df_results = pd.DataFrame(dict_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    
    ax = df_results.plot.box(vert=False, whis=10)
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    
    plt.show()

    return df_results