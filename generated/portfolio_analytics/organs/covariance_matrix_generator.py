import numpy as np

def covariance_matrix_generator(data):
    """
    Generates covariance and correlation matrices from given data.

    Parameters:
    data (list of lists): A list of observations, each observation being a list of values.

    Returns:
    dict: Contains covariance matrix and correlation matrix.
    """
    # Convert the data to a numpy array
    data_array = np.array(data)

    # Calculate the covariance matrix
    cov_matrix = np.cov(data_array, rowvar=False)

    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data_array, rowvar=False)

    # Return the results
    return {
        "covariance_matrix": cov_matrix.tolist(),
        "correlation_matrix": corr_matrix.tolist()
    }
