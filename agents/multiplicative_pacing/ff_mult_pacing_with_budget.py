from agents.hedge.hedge import HedgeAgent
import numpy as np

class FFMultiplicativePacingAgent:
    def __init__(self, bids_set, valuations, budget, T, eta):
        self.bids_set = bids_set
        self.K = len(bids_set)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K)/T))
        self.valuations = valuations
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0
        self.budget_depleted = False

    def bid(self):
        if self.budget < 1:
            self.budget_depleted = True
            return -1 # ---> never winning the auction
        return self.bids_set[self.hedge.pull_arm()]

    def update(self, f_t, c_t, m_t):
        if self.budget_depleted == False:
          # update hedge
          f_t_full = np.array([(self.valuations[0]-b)*int(b >= m_t) for b in self.bids_set])
          c_t_full = np.array([b*int(b >= m_t) for b in self.bids_set])
          L = f_t_full - self.lmbd*(c_t_full-self.rho)
          range_L = 2+(1-self.rho)/self.rho
          self.hedge.update((2-L)/range_L) # hedge needs losses in [0,1]
          # update lagrangian multiplier
          self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), a_min=0, a_max=1/self.rho)
          # update budget
          self.budget -= c_t
        else:
          pass

    def get_budget_depleted(self):
      return self.budget_depleted