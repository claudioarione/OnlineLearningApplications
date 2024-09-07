from utils.util_functions import compute_weights
import numpy as np

class UCB1Agent:
    def __init__(self, bids_set,valuations, budget, T, range=1):

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
        self.valuations = valuations
        self.rho = self.budget/self.T

        self.budget_depleted = False

    def bid(self):
      if self.budget < 1:
            self.budget_depleted = True
            return -1 # ---> never winning the auction
      else:
        if self.t < self.K:
            self.a_t = (self.t) / (self.K-1) #rescaled bid
        else:
            ucbs_rewards = self.average_rewards + self.range*np.sqrt(2*np.log(self.t)/self.N_pulls) #can also use self.T instead of self.t
            ucbs_costs = self.average_costs - self.range*np.sqrt(2*np.log(self.t)/self.N_pulls) #can also use self.T instead of self.t

            weights = compute_weights(self.rho, ucbs_rewards, ucbs_costs) #Compute the weights using linear program
            self.a_t = np.random.choice(np.arange(self.K), p=weights) / (self.K-1) #Sample an arm from the distribution obtained through weights
        return self.a_t

    def update(self, r_t, c_t):
        if self.budget_depleted == False:
          index = int(self.a_t*(self.K-1))

          self.N_pulls[index] += 1
          self.average_rewards[index] += (r_t - self.average_rewards[index])/self.N_pulls[index]
          self.average_costs[index] += (c_t - self.average_costs[index])/self.N_pulls[index]

          self.t += 1
          self.budget -= c_t
        else:
          pass

    def get_budget_depleted(self):
      return self.budget_depleted