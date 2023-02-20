import numpy as np
#from sklearn.metrics import mean_squared_error

# y_std = np.var(dataTest, ddof=1)
def evaluation(errorType, y_true, y_pred, y_std):

    if errorType == "NMRSE":
        # Normalized Root Mean Squared Error
        print("NRMSE")
        error = np.sqrt(mean_squared_error(y_true, y_pred)) / y_std

    elif errorType == "NMSE":
        # Normalized Mean Squared Error
        error = (mean_squared_error(y_true, y_pred)) / y_std
    else:
        # By default return: Mean Squared Error
        error = mean_squared_error(y_true, y_pred)

    return error