def calc_trix(df, periods=14, close_col='close', drop_ema=False):

    """
    Function to calculate Triple Exponential Average (TRIX)

    Source: https://www.investopedia.com/terms/t/trix.asp
        
    Parameters:
    -----------
        df : DataFrame
            pandas DataFrame
        
        periods : int
            The period over which to calculate the indicator value
        
        close_col : str 
            The name of the CLOSE values column
        
    Returns
    -------
        df : DataFrame
            Copy of 'data' DataFrame with 'trix' columns added
    """
    
    df['trix_ema1'] = df[close_col].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    df['trix_ema2'] = df['trix_ema1'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    df['trix_ema3'] = df['trix_ema2'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()

    df['trix_ema3_previous'] = df['trix_ema3'].shift(1)

    df['trix'] = (df['trix_ema3'] - df['trix_ema3_previous']) / df['trix_ema3_previous']

    df.drop(['trix_ema3_previous'], axis=1, inplace=True)

    if drop_ema:

        df.drop(['trix_ema1', 'trix_ema2', 'trix_ema3'], axis=1, inplace=True)
        
    return df
