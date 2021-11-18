import numpy as np


def p(prices_historical=None, demand_historical=None, information_dump=None):
    """
    this pricing algorithm would return the minimum price used
    by any competitor in the last iteration, it returns a random
    price if it is the first iteration

    input:
    prices_historical: numpy 2-dim array: (number competitors) x (past iterations)
    it contains the past prices of each competitor (you are at index 0) over the past iterations

    demand_historical: numpy 1-dim array: (past iterations)
    it contains the history of your own past observed demand over the last iterations

    information_dump: some information object you like to pass to yourself at the next iteration
    """
    if demand_historical is None:
        return round(np.random.uniform(30, 80), 1), None
    next_price = np.min(prices_historical[1:, -1])
    return round(next_price, 1), information_dump


