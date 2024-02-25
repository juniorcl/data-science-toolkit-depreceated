import pandas as pd


def top_k_recall(y_true, y_score, k=10):
    
    df = pd.DataFrame({'true': y_true, 'score': y_score[:, 1]})
    
    df.sort_values('score', ascending=False, inplace=True)

    df = df.reset_index(drop=True, inplace=True)
    
    df['ranking'] = df.index + 1 

    df['recall_top_k'] = df['true'].cumsum() / df['true'].sum()

    return df.loc[k, 'recall_top_k']