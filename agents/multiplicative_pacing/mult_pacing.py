import numpy as np

# Truthful Second Price Auction Agent definition
class MultiplicativePacingAgent:
    def __init__(self, valuation, budget, T, eta, bids_set=None):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.bids_set = bids_set
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        con_bid = self.valuation/(self.lmbd+1) # Scaled bid
        if self.bids_set is None:
            return con_bid
        return min(self.bids_set, key=lambda x:abs(x-con_bid))

    def update(self, f_t, c_t):
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t),
                            a_min=0, a_max=1/self.rho)
        #Update the lambda
        self.budget -= c_t