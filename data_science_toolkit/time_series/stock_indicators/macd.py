from .ewm import calc_ewm


def calc_macd(df, period_long=26, period_short=12, period_signal=9, column='close', drop_ewm=False):

    """
    Function to calculate the Moving Average Convergence Divergence (MACD) indicator and Signal Line.

    Source: https://www.quantifiedstrategies.com/python-and-macd-trading-strategy/

    Parameters
    ----------
    df : DataFrame
        Data frame with the time series.

    period_long: int
        Period long to calculate the Exponentially Weighted Mean (EWM).

    period_short: int
        Period short to calculate the EWM.

    period_signal: int
        Period signal to calculate the EWM.

    column : str
        The name of the variable to calculate the Bollinger Bands.

    drop_ewm : Bool
        Parameter to define whether the column EWM will be droped.
    
    Returns
    -------
    df : DataFrame
        Data Frame with all the variables and Bollinger Bands variables.
    """
    
    df = calc_ewm(df, period_long, column=column)

    df = calc_ewm(df, period_short, column=column)

    df['macd'] = df[f'ewm_{period_short}'] - df[f'ewm_{period_long}']

    df['macd_signal'] = df['macd'].ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()

    if drop_ewm:
        
        df = df.drop([f'ewm_{period_short}', f'ewm_{period_long}'], axis=1)

    return df