from dprl.actor import p as p0
from dprl.others import random
from dprl.main import main

if __name__ == '__main__':
    competitors_f = [p0, random, random, random, random]
    price_matrix, sales_matrix = main(competitors_f)