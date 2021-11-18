import numpy as np
import math

# Number of rounds
T = 1000


def market2(prices):
    # As in Group Project: Dynamic Pricing under Competition
    # by Xinzhuo Jiang, Xiao Lei, Ting Liu, Wenyu Ma, Shixiang Zhou
    exp_price = list(map(lambda x: np.exp(-x), prices))
    sum_price = sum(exp_price)
    prob_list = list(map(lambda x: x / sum_price, exp_price))

    # generate a total demand
    D = 4000 + np.random.normal(1000, 500, 1)
    sales = list(map(lambda x: math.floor(x * D), prob_list))
    return sales


def market(prices):
    """ Calcola i sales come inversamente proporzionali ai prezzi """
    tot = 1000
    n = len(prices)
    sales = [round(tot/p/n) for p in prices]
    return sales


def main(competitors_funcs):
    """
    Ciclo principale
    :param competitors_funcs: lista di funzioni dei competitors
    """

    #number of teams
    n = len(competitors_funcs)

    #initial price and demand matrix
    price_matrix = np.empty([n,T])
    price_matrix[0:n, 0:T] = np.NAN
    sales_matrix = np.empty([n,T])
    sales_matrix[:] = np.NAN
    argument_list = [None]*len(competitors_funcs)


    #iterate over T
    for t in range(T):
        if t == 0:
            prices = list(map(lambda x: x(None, None), competitors_funcs))
        else:
            #former price matrix
            former_price = price_matrix[0:n, 0:t]
            #compute the price at time i
            prices = list(map(lambda x, y: x(former_price, sales_matrix[competitors_funcs.index(x), 0:t], y), competitors_funcs, argument_list))
        argument_list = [j for i, j in prices]
        prices = [i for i, j in prices]

        # Market simulator
        sales = market(prices)
        print("Round " + str(t))
        print("Prices: " + str(prices))
        print("Sales: " + str(sales))

        #update matrixs
        price_matrix[:,t] = prices
        sales_matrix[:,t] = sales

    return price_matrix, sales_matrix

