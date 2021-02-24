import numpy  as np
import pandas as pd



def top_k_precision(y_true, y_score, k=10):
    
    df = pd.DataFrame({'true': y_true, 'score': y_score[:, 1]})
    df.sort_values('score', ascending=False, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    df['ranking'] = df.index + 1

    df['precision_top_k'] = df['true'].cumsum()/df['ranking'] 

    return df.loc[k, 'precision_top_k']


def top_k_recall(y_true, y_score, k=10):
    
    df = pd.DataFrame({'true': y_true, 'score': y_score[:, 1]})
    df.sort_values('score', ascending=False, inplace=True)

    df = df.reset_index(drop=True, inplace=True)
    df['ranking'] = df.index + 1 

    df['recall_top_k'] = df['true'].cumsum()/df['true'].sum()

    return df.loc[k, 'recall_top_k']


def top_k_f1score(y_true, y_score, k=10):

    precision = top_k_precision(y_true, y_score, k)
    recall = top_k_recall(y_true, y_score, k)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def top_k_scores(model_name, y_true, y_score, k=10):

    p = top_k_precision(y_true, y_score, k)
    r = top_k_recall(y_true, y_score, k)
    f1 = top_k_f1score(y_true, y_score, k)

    df = pd.DataFrame({
        'precision_top_k': np.round(p, 4),
        'recall_top_k': np.round(r, 4),
        'f1_top_k': np.round(f1, 4)
    }, index=model_name)

    return df