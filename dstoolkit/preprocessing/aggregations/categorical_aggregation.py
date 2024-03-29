def mode(arr):

    """
    Returns the most frequent value in a numpy array.
    """
    
    arr = list(arr)
    
    return max(set(arr), key=arr.count)


def agg_cat(df, groupby, variables):

    """
    Function to calculate nunique and mode from a Series

    Parameters
    ----------
    df : DataFrame
        Initial DataFrame

    groupby : List
        List of variables to be groupped.

    variables : List
        List of variables to apply the functions.
    
    Returns
    -------
    df_agg : DataFrame
        Result DataFrame. 
    """

    list_funcs = ['nunique', mode]

    dict_funcs = {var: list_funcs for var in variables}

    df_agg = df.groupby(groupby).agg(dict_funcs)

    df_agg.columns = [f'{col[0]}_{col[1]}' for col in df_agg.columns]

    df_agg = df_agg.reset_index()

    return df_agg