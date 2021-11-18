import numpy as np


def random(prices_historical=None, demand_historical=None, information_dump=None):
    return round(np.random.uniform(1, 100), 0), None