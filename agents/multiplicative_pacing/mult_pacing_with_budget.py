
import numpy as np

class MultiplicativePacingAgent:
    def __init__(self, bids_set, valuations, budget, T, eta):
        self.bids_set = bids_set #Need to discretize the bids
        self.valuations = valuations
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 0
        self.t = 0
        self.budget_depleted = False

    def bid(self):
        if self.budget < 1:
            self.budget_depleted = True
            return -1 # ---> never winning the auction

        con_bid = self.valuations[0]/(self.lmbd+1) # Scaled continuos bid
        #return con_bid

        #Discrete bid: since other regret minimizers have discrete set of bid, we impose this also to MPAgent by choosing the closest bid
        return min(self.bids_set, key=lambda x:abs(x-con_bid))

    def update(self, f_t, c_t):
        if self.budget_depleted == False:
          self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t),
                              a_min=0, a_max=1/self.rho)
          #Update the lambda
          self.budget -= c_t
        else:
          pass

    def get_budget_depleted(self):
      return self.budget_depleted