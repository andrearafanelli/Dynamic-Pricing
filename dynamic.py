import numpy as np
from sklearn import linear_model
from scipy.stats import poisson
from sklearn.externals._arff import xrange



def all_features(x0, x1, x2, x3, x4):
    return np.array([x0] + [x1 ** 3] + [x2 ** 3] + [x3 ** 3]+[x4 ** 3]+[x2*x3]+[x3*x4]+[x2*x4])

def model_demand_learning(market_learning, price_learning, sales_data_learning):
    features_model = all_features(np.ones(market_learning.shape[1]), price_learning, market_learning.mean(axis=0),
                                 np.amin(market_learning, axis=0),np.amax(market_learning,axis=0))
    model = linear_model.LinearRegression().fit(features_model.T, sales_data_learning)

    return model


def demand_learning(model, feats, sales):
    expected_sales = model.predict(feats.reshape(1, -1))

    probability = float(poisson.pmf(sales, expected_sales))
    return probability, expected_sales

def probabilities(expected_sales,sales):
   return float(poisson.pmf(sales, expected_sales))



class MarkovDecisionProcess:
    def __init__(self, transition={}, reward={}, gamma=.7):
        self.states = transition.keys()
        self.transition = transition
        self.reward = reward
        self.gamma = gamma

    def actions(self, state):
        return self.transition[state].keys()

    def R(self, state, action):
        return self.reward[state][action]

    def T(self, state, action):
        return self.transition[state][action]


def value_iteration(mdp, gamma, epsilon):
    states = mdp.states
    actions = mdp.actions
    T = mdp.T
    R = mdp.R
    V1 = {s: 0 for s in states}
    while True:
        V = V1.copy()
        delta = 0
        for s in states:
            for a in actions(s):
                V1[s] = R(s, a) + gamma * max([sum([p * V[s1] for (p, s1) in T(s, a)])])
            delta = max(delta, abs(V1[s] - V[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return V


def best_policy(V, mdp):
    states = mdp.states
    actions = mdp.actions
    pi = {}
    for s in states:
        pi= max(actions(s), key=lambda a:expected_utility(a, s, V, mdp))
    return pi


def expected_utility(a, s, V, mdp):
    T = mdp.T
    return sum([p * V[s1] for (p, s1) in mdp.T(s, a)])


def p(prices_historical=None, demand_historical=None, information_dump=None):
    if (prices_historical is None and demand_historical is None):
        information_dump = {
            "Number of Competitors": None,
            "Actual period": 0,
            "Reward": {},
            "Transition": {},
            "Probability": []

        }
        #return (np.random.randint(50,90), information_dump)
        return (np.random.uniform(1,100), information_dump)
    period = int(np.array(prices_historical.shape[1]))
    if period < 150:
        #return (np.random.randint(50,90), information_dump)
        return (np.random.uniform(1, 100), information_dump)
    information_dump["Actual Period"] = period
    competitor_price = prices_historical[1:, :]
    competitors = prices_historical.shape[0] - 1
    information_dump["Number of competitors"] = competitors
    mean_competitor_price = float(np.mean(competitor_price[:, period - 1]))
    min_competitor_price = float(min(competitor_price[:, period - 1]))
    max_competitor_price=float(max(competitor_price[:,period-1]))
    market = np.array(competitor_price[:, :-1])
    price = prices_historical[0, :-1]
    sales_data = demand_historical[:-1]
    regression = model_demand_learning(market, price, sales_data)
    action=np.random.choice(price,20,replace=False)





    for i in xrange (0,101):
        Transition = {}
        Reward = {}
        Final_state = {}
        Final_reward = {}

        for j in action:
            features = all_features(1, j, mean_competitor_price, min_competitor_price,max_competitor_price)
            probability, expected_sales = demand_learning(regression, features,i)

            Transition[j] = [(1-probability, i+1)]
            Reward[j] = int(j * expected_sales)
            Final_state[j] = [(0, 100)] #Terminal state
            Final_reward[j] = 0 #Terminal state
        if i!=(100):
            information_dump['Transition'][i] = Transition
            information_dump['Reward'][i] = Reward
        else:
            information_dump['Transition'][i] = Final_state
            information_dump['Reward'][i] = Final_reward


    mdp = MarkovDecisionProcess(transition=information_dump['Transition'], reward=information_dump['Reward'])
    gamma = 0.7
    epsilon = 0.05
    value_function = value_iteration(mdp, gamma, epsilon)

    price=best_policy(value_function,mdp)


    return (price, information_dump)
