import numpy as np
from utils.util_functions import compute_weights

class UCB1Agent:
    def __init__(self, bids_set,valuation, budget, T, range=1):

        self.bids_set = bids_set
        self.K = len(bids_set)
        self.T = T
        self.range = range

        self.a_t = None
        self.average_rewards = np.zeros(self.K)
        self.average_costs = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        self.t = 0

        self.budget = budget
        self.valuation = valuation
        self.rho = self.budget/self.T

    def bid(self):
      if self.budget < 1:
            return 0
      else:
        if self.t < self.K:
            self.a_t = (self.t) / (self.K-1) #rescaled bid
        else:
            ucbs_rewards = self.average_rewards + self.range*np.sqrt(2*np.log(self.t)/self.N_pulls) #can also use self.T instead of self.t
            ucbs_costs = self.average_costs - self.range*np.sqrt(2*np.log(self.t)/self.N_pulls) #can also use self.T instead of self.t

            weights = compute_weights(self.rho, ucbs_rewards, ucbs_costs)
            self.a_t = np.random.choice(np.arange(self.K), p=weights) / (self.K-1)

            #self.a_t = np.argmax(ucbs_rewards) / self.K
        return self.a_t

    def update(self, r_t, c_t):
        index = int(self.a_t*(self.K-1))

        self.N_pulls[index] += 1
        self.average_rewards[index] += (r_t - self.average_rewards[index])/self.N_pulls[index]
        self.average_costs[index] += (c_t - self.average_costs[index])/self.N_pulls[index]

        self.t += 1
        self.budget -= c_t