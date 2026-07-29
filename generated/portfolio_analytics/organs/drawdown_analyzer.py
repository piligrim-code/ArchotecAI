import math

def drawdown_analyzer(cumulative_returns):
    """
    Analyze the cumulative returns to determine the maximum drawdown.

    :param cumulative_returns: List of cumulative returns over time
    :return: Dictionary containing the maximum drawdown and the peak/trough points
    """
    max_drawdown = 0
    peak = 0
    trough = 0

    for i, ret in enumerate(cumulative_returns):
        if ret > peak:
            peak = ret
            max_drawdown = 0

        drawdown = (peak - ret) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            trough = ret

    return {
        "max_drawdown": max_drawdown,
        "peak": peak,
        "trough": trough
    }
