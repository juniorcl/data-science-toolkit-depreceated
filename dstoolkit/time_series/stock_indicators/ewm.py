def calc_ewm(df, period=0, column='close'):
    
    """
    Function to calculate the Exponentially Weighted Mean (EWM).

    parameters
    ----------
    df : DataFrame
        Data frame with the time series.

    period : int
        Period to calculate the EWM.

    column : str
        The name of the variable to calculate the Exponentially Weighted Mean.

    return
    ------
    df : DataFrame
        Data Frame with all the variables and EWM variable.
    """

    df[f'emw_{period}'] = df[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()    
    
    return df
