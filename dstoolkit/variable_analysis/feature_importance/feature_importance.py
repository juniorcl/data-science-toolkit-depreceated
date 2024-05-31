import pandas  as pd
import seaborn as sns

import matplotlib.pyplot as plt


def get_tree_feature_importance(model, X_train):
    
    df_imp = pd.DataFrame(model.feature_importances_, X_train.columns).reset_index()
    df_imp.columns = ["Variable", "Importance"]
    df_imp = df_imp.sort_values("Importance", ascending=False)
    
    sns.barplot(x="Importance", y="Variable", color="#006e9cff", data=df_imp[:20])
    
    plt.title(f"Importance of Variables")
    plt.show()

    return df_imp