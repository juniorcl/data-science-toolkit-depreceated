import pandas as pd


def top_k_precision(y_true, y_score, k=10):
    
    df = pd.DataFrame({'true': y_true, 'score': y_score[:, 1]})
    df.sort_values('score', ascending=False, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    df['ranking'] = df.index + 1

    df['precision_top_k'] = df['true'].cumsum()/df['ranking'] 

    return df.loc[k, 'precision_top_k']