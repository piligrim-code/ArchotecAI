import math
import statistics

def volatility_calculator(prices):
    """
    Calculate the annual volatility and Sharpe ratio for each asset given a list of prices.

    :param prices: List of historical prices for an asset.
    :return: A dictionary containing the annual volatility and Sharpe ratio.
    """
    if not prices:
        raise ValueError("Prices list cannot be empty")

    returns = [prices[i+1] / prices[i] - 1 for i in range(len(prices)-1)]
    mean_return = statistics.mean(returns)
    variance = statistics.variance(returns)
    volatility = math.sqrt(variance) * math.sqrt(252)  # Assuming 252 trading days in a year

    # Assuming a risk-free rate of 0 for simplicity
    sharpe_ratio = mean_return / volatility if volatility != 0 else float('nan')

    return {
        'annual_volatility': round(volatility, 4),
        'sharpe_ratio': round(sharpe_ratio, 4)
    }

async def volatility_calculator_async(prices):
    return volatility_calculator(prices)

# Example usage:
# prices = [100, 105, 103, 104, 106]
# print(volatility_calculator(prices))
