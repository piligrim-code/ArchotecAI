import numpy as np

def min_variance_optimizer(cov_matrix, returns=None):
    """
    Compute weights of minimum variance portfolio using numpy.linalg.

    :param cov_matrix: Covariance matrix of asset returns
    :param returns: Expected returns of assets
    :return: Dictionary containing weights of minimum variance portfolio and its variance
    """
    if not isinstance(cov_matrix, np.ndarray) or cov_matrix.ndim != 2:
        raise ValueError("Covariance matrix must be a 2D numpy array.")

    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if returns is None:
        returns = np.ones(cov_matrix.shape[0])

    normalized_returns = returns / np.sum(returns)

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights = np.dot(inv_cov_matrix, normalized_returns)
    weights /= np.sum(weights)

    portfolio_variance = np.dot(np.dot(weights.T, cov_matrix), weights)

    return {
        "weights": weights.tolist(),
        "portfolio_variance": portfolio_variance.item()
    }
