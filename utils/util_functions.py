from scipy.optimize import linprog
import numpy as np

##### Utility functions for the project #####

def rescale(x, min_x, max_x):
    """
    Rescale a value x from the interval [0, 1] to the interval [min_x, max_x]
    """
    return min_x + (max_x-min_x)*x

def compute_m_t(element,bids):
  temp = bids.copy()
  temp.remove(element)
  temp.sort()
  return temp[-1]

def compute_weights(rho, rewards, costs):
  """
  Compute the weights for the rewards in the objective function of the optimization problem
  """
  l = len(rewards)
  c = (-rewards).tolist() # declare coefficients of the objective function
  A_in = [costs.tolist()] # declare the inequality constraint matrix
  b_in = [rho] # declare the inequality constraint vector
  A_e = [np.ones(l)] # declare the equality constraint matrix
  b_e = [1] # declare the equality constraint vector
  results = linprog(c=c, A_ub=A_in, b_ub=b_in, A_eq=A_e, b_eq=b_e, method='highs-ds')   # solve
  return results.x

def extract(pos, lst):
  return [item[pos] for item in lst]

def extract_winners(placement, p, lst):
  p_history = []
  for item in lst:
    if p in item and item.index(p) == (placement-1): #Placement = 1 is first place, Placement = 2 is second place
      p_history.append(1)
    else:
      p_history.append(0)
  return p_history