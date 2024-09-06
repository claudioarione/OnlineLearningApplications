import numpy as np

class MultiProductPricingEnv:
    def __init__(self, conversion_probabilities, costs):
        self.conversion_probabilities = conversion_probabilities
        self.costs = costs

    def round(self, p_t, n_t):
        # FIXME: conversion_probabilities was taking the global variable, but it should be self.conversion_probabilities
        d_t = np.array(len(self.conversion_probabilities))
        r_t = np.array(len(self.conversion_probabilities))
        for i in range(len(self.conversion_probabilities)):
            conv_prob = self.conversion_probabilities[i]
            print(p_t[0], p_t[1])
            print(conv_prob(p_t[0], p_t[1]))
            d_t[i] = np.random.binomial(n_t, conv_prob(p_t[0], p_t[1]))
            r_t[i] = (p_t - self.costs[i])*d_t[i]
        return d_t, r_t