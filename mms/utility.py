"""
Author: Zirou Qiu
Last modfied: 10/16/2020 
Description: 
    This module consists of utility funcitons
"""

import numpy as np


# Check if a matrix is symmetric
def is_symmetric(A, rtol=1e-05, atol=1e-08):
    # Source: https://stackoverflow.com/a/42913743
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

# Generate a ER_newtwork and return the adjacency matrix 
def ER_network(n, p):
   M = np.random.choice([0, 1], size = (n, n), p=[1.0-p, p])
   return M

