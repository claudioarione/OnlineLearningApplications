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
        self.N_pulls = np.zeros(self.K)
        self.t = 0

        self.budget = budget
        self.valuation = valuation

    def bid(self):
      if self.budget < 1:
            return 0
      else:
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs_rewards = self.average_rewards + self.range*np.sqrt(2*np.log(self.t)/self.N_pulls) #can also use self.T instead of self.t
            self.a_t = np.argmax(ucbs_rewards)
        return self.bids_set[self.a_t]

    def update(self, r_t, c_t):
        index = self.a_t

        self.N_pulls[index] += 1
        self.average_rewards[index] += (r_t - self.average_rewards[index])/self.N_pulls[index]

        self.t += 1
        self.budget -= c_t