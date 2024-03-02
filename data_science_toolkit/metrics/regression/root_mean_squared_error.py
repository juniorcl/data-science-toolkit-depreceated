from numpy import sqrt
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred):

    """
    Function to calulate root mean squared error.

    Parameters
    ----------
    y_true : array-like or Series
        Target values.

    y_pred : array-like or Series
        predicted values.

    Returns
    -------
    rmse : float
        A non-negative floating point value.
    """

    mse = mean_squared_error(y_true, y_pred)

    rmse = sqrt(mse)

    return rmse