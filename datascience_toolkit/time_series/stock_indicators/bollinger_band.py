def calc_bollinger_band(df, column='adj_close', n=20, k=2):
    
    """
    Create a new variable with the bollinger bands:
        - Upper Bollinger Band;
        - Middle Bollinger Band;
        - Lower Bollinger Band.

    Parameters
    ----------
    df : DataFrame
        Data frame with the time series.

    column : str
        The name of the variable to calculate the Bollinger Bands.

    n : int
        Number of periods to calculate the moving average.

    k : int
        Number of k times the standard deviation of asset prices.
    
    Returns
    -------
    df : DataFrame
        Data Frame with all the variables and Bollinger Bands variables.
    """

    df['standard_desviation_bollinger_band'] = df[column].rolling(n).std()

    df['middle_bolling_band'] = df[column].rolling(n).mean()

    df['upper_bolling_band'] = df['middle_bolling_band'] + df['standard_desviation_bollinger_band'] * k

    df['lower_bolling_band'] = df['middle_bolling_band'] - df['standard_desviation_bollinger_band'] * k

    df = df.drop(['standard_desviation_bollinger_band'], axis=1)

    return df