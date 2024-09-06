import numpy as np

# Pricing Environment definition
class NonStationaryPricingEnv:
    def __init__(self, mu, cost):
        self.mu = mu
        self.cost = cost

    def round(self, p_t, n_t, t):
        conv_prob = self.mu[t]
        noise = int(np.random.normal(0, 4))
        d_t = n_t * conv_prob(p_t) + noise
        r_t = (p_t - self.cost)*d_t
        return d_t, r_t