"""
Author: Zirou Qiu
Last modfied: 10/31/2020
Description: 
    This module consists of simulations of the spread of multiple 
    contigions on a single network under the threshold model.
"""


#--------------------------- Imports ------------------------------#
import numpy as np
from guppy import hpy
import mms.utility as mu
from scipy import sparse

#----------------------- Funciton Defintions ----------------------#
def isolate_threshold_count(A, B, T, k, r = 0):
    """
        Description
        -----------
        This function simulate the spread of multiple contigions on a single network
        where each contagion has 2 states (0 or 1).
        Contagions are not interrelated.

        Parameters
        ----------
        A: scipy array, int {0, 1}
            The adjacency matrix of G.
            A is sparse

        B: scipy array, int {0, 1}
            The initial configuration matrix where $B_{vj}$ is the state value of 
            vertex v for contagion j.
            B is sparse

        T: numpy array, int
            The threshold matrix where $T_{vj}$ is the threshold of vertex v for
            contagion j.

        k: int
            The number of system iterations

        r: float, optional
            The recovery probability. In each iteration, each vertex has a probability 
            r changing the state to 0 for each contigion.

        Returns
        -------
        B: numpy array
            The final configuration
    """

    # Make all 1s along the diagonal of A (since we are considering the closed neighborhood)
    #np.fill_diagonal(A, 1)

    # The recovery probability
    recovery = False
    if r != 0:
        recovery = True

    # The main loop
    for i in range(k):
        # matrix operation
        B_last = B
        B = A @ B - T #B = np.matmul(A, B_last) - T

        # update states
        B[B >= 0] = 1
        B[B < 0] = 0

        # If a recovery probability is set
        if recovery:
            B[np.random.rand(*B.shape) < r] = 0

        # if fixed point
        if np.array_equal(B, B_last):
            print("A fixed point is reached at iteration {}".format(i))
            return B

    print("Max number of iteratios reached")
    return B

###########################################################################################

def correlate_threshold_weight(A, B, T, W, k, r = 0):
    """
        Description
        -----------
        This function simulate the spread of multiple contigions on a single network
        where each contagion has 2 states (0 or 1).
        Contagions are interrelated as described by the thrid model.

        Parameters
        ----------
        A: numpy array, int {0, 1}
            The adjacency matrix of G.
            A is sparse

        B: numpy array, int {0, 1}
            The initial configuration matrix where $B_{vj}$ is the state value of 
            vertex v for contagion j.
            B is sparse 

        T: numpy array, int
            The threshold matrix where $T_{vj}$ is the threshold of vertex v for
            contagion j.

        W: numpy array, float [0, 1]
            The weight matrix where $W_{ij}$ is the weight of contagion j w.r.t 
            contagion i

        k: int
            The number of system iterations

        r: float, optional
            The recovery probability. In each iteration, each vertex has a probability 
            r changing the state to 0 for each contigion.

        Returns
        -------
        B: numpy array
            The final configuration

    """

    # Make all 1s along the diagonal of A (since we are considering the closed neighborhood)
    #A.setdiag(1)

    # The recovery probability
    recovery = False
    if r != 0:
        recovery = True

    # Take the transpose of the weight matrix
    W = np.transpose(W)

    # The main loop
    for i in range(k):
        # matrix operation
        B_last = B
        #B = np.linalg.multi_dot([A, B_last, W]) - T 
        B = A @ B_last @ W - T

        # update states
        B[B >= 0] = 1
        B[B < 0] = 0

        # If a recovery probability is set
        if recovery:
            B[np.random.rand(*B.shape) < r] = 0

        # if fixed point
        if np.array_equal(B, B_last):
            print("A fixed point is reached at iteration {}".format(i))
            return B
    
    #h = hpy()
    #print(h.heap())

    print("Max number of iteratios reached")
    return B


def correlate_threshold_density(A, B, T, d, k):
    """
        Description
        -----------
        This function simulate the spread of multiple contigions on a single network
        where each contagion has 2 states (0 or 1).
        Contagions interrelated as described by the second model.

        Parameters
        ----------
        A: numpy array, int {0, 1}
            The adjacency matrix of G.
            A is sparse

        B: numpy array, int {0, 1}
            The initial configuration matrix where $B_{vj}$ is the state value of 
            vertex v for contagion j.
            B is sparse

        T: numpy array, int
            The threshold matrix where $T_{vj}$ is the threshold of vertex v for
            contagion j.

        d: numpy array, int
            The density vector

        k: int
            The number of system iterations

        Returns
        -------
        B: numpy array
            The final configuration

    """
    # Compute the reciprocal 
    d_bar = np.transpose( np.reciprocal(d.astype(float)) ) # Make sure that d is a column vector

    # The number of contagions
    c =  np.shape(T)[1]

    # k * 1 ones
    one = np.ones((c, 1), dtype = 'float')
    
    # The main loop
    for i in range(k):
        B_last = B

        # Compute M
        M = B @ one @ d_bar #M = np.linalg.multi_dot([B, one, d_bar]) 
        M[M >= 1.0] = 1.0
        M[M < 1.0] = 0.0

        #B = np.matmul(A, M) - T
        B = A @ M - T

        # update states
        B[B >= 0.0] = 1.0
        B[B < 0.0] = 0.0

        # if fixed point
        if np.array_equal(B, B_last):
            print("A fixed point is reached at iteration {}".format(i))
            return B

    print("Max number of iteratios reached")
    return B

def two_layer_1(A, D_inverse, a_1, t_2, t_3, b_1, b_2, p, alpha, beta, r, k):
    """
        Description
        -----------
        This funciton simulate the spread of two contagions on two different
        networks, where contagions are correlated as described in the project
        report. 
        
        Parameters
        ----------
        A: n x n scipy sparse matrix, int {0, 1}
           The adjacency matrix of G
        
        D_inverse: n x n scipy sparse matrix, float [0, 1]
            The inversed diagonal matrix of G
        
        a_1: n x 1 scipy sparse matrix, int {0, 1}
            If a person is prosocial or not
    """
    
    # Compute the degree fraction matrix
    F = D_inverse @ A
    F = sparse.csr_matrix(F)

    # The number of vertices
    n = np.shape(A)[0]
    
    # The one vector
    one = np.ones((n, 1), dtype = 'float')
    zero = np.zeros((n, 1), dtype = 'float')

    # The initial recovery vector (no one has recovered)
    b_3 = np.zeros((n, 1), dtype = 'float')
    b_3 = sparse.csr_matrix(b_3)

    # The inital susceptible vector
    b_4 = -b_2 - b_3
    b_4[b_4 == 0.0] = 1.0
    b_4[b_4 < 0.0] = 0.0

    # The main loop
    for i in range(k):
        # need b_1__last to Update the state of the second contagion
        b_1_last = b_1 
        
        # Deteremine if a fixed point is reached, that is everyone has recovered (if r != 0)
        b_4_last = b_4 
        
        # This can also be used to determine if a fixed point is reached
        b_2_last = b_2
    
        # The fraction of total number of infections
        a_3 = np.count_nonzero(b_2) / float(n)
        
        # determine if the overall faction of infection exceed the threshold
        l_3 = -(t_3 - a_3) # Note that I cannot do a_3 - t_3 since a_3 is not a vector
        l_3[l_3 >= 0.0] = 1.0
        l_3[l_3 < 0.0] = 0.0
        #l3 = t_3 <= a_3
        
        # Determine if the fraction of neighbors with wear face masks exceeds a threshold
        l_2 = F @ b_1_last - t_2 # sparse? 
        l_2[l_2 >= 0.0] = 1.0
        l_2[l_2 < 0.0] = 0.0
        #l_2 = (F @ b_1_last) >= t_2 WORTH TRYING!
        
        # Update the mask state b_1
        b_1 = a_1 + l_2 + l_3 # logical operation?
        b_1[b_1 >= 1.0] = 1.0 # sparse?
        #b_1 = np.logical_or(np.logical_or(a_1, l_2), l_3) WORTH TRYING

        # The # of infected neighbors of each v
        d = A @ b_2_last
        
        # The # of infected neighbors with mask
        d_2 = A @ np.multiply(b_1_last, b_2_last) # Very important to pass b_1_last here

        # The # of infected neighbors without mask
        d_1 = d - d_2
        
        #Only susceptibles (b_4) can be infected 
        #--------------------------------------------------#
        # h1 : the probability of not getting infected from neighbors who do not wear masks (1 - p or 1 - alpha p)
        temp = one - (b_1 * (1.0 - alpha)) # IMPORTANT!!!!! b_1_last? (syn vs asyn)
        h_1 = one - (temp * p)
            
        # h2: contains the probability of not getting infected from neighbors who wear masks (1 - beta p or 1 - alpha beta p)
        h_2 = one - (temp * beta * p)

        temp = np.multiply(np.power(h_1, d_1), np.power(h_2, d_2))
        q = np.multiply(b_4, one - temp)
        #--------------------------------------------------#
        
        # Has to flatten q to pass it to the binomial funciton
        q_f = q.flatten()
        
        # Compute newly infected nodes
        newly_infected = np.reshape(np.random.binomial(1, q_f), (-1 ,1))
        
        # Recovery
        rr = np.random.choice([0, 1], size = (n, 1), p=[1.0 - r, r])
        b_3 = np.logical_and(b_2, rr) + b_3 # update b_3
        b_2 = b_2 - rr
        b_2[b_2 == -1] = 0.0
        
        # Update b_2
        b_2 = newly_infected + b_2
        
        # Update the susceptible vector
        b_4 = -b_2 - b_3
        b_4[b_4 == 0.0] = 1.0
        b_4[b_4 < 0.0] = 0.0
        
        # Determine if a fixed point it reached
        if np.array_equal(b_2, b_2_last):
#             print("A fixed point is reached at iteration {}".format(i))
#             return b_2
            return np.count_nonzero(b_2) - 1 ####################

#     print("Maximum number of iteration is reached")    
#     return b_2
    return np.count_nonzero(b_2) - 1 ####################


# The SIS model, and each node recoverys in exact one day (r = 1)
def two_layer_SIS(A, D_inverse, a_1, t_2, t_3, b_1, b_2, p, alpha, beta, r, k):
    # Compute the degree fraction matrix
    F = D_inverse @ A
    F = sparse.csr_matrix(F)

    # The number of vertices
    n = np.shape(A)[0]
    
    # The one vector
    one = np.ones((n, 1), dtype = 'float')
    zero = np.zeros((n, 1), dtype = 'float')

    # The initial recovery vector (no one has recovered)
    b_3 = np.zeros((n, 1), dtype = 'float')
    b_3 = sparse.csr_matrix(b_3)

    # The inital susceptible vector
    b_4 = -b_2 - b_3
    b_4[b_4 == 0.0] = 1.0
    b_4[b_4 < 0.0] = 0.0

    # The main loop
    for i in range(k):
        # need b_1__last to Update the state of the second contagion
        b_1_last = b_1 
        
        # Deteremine if a fixed point is reached, that is everyone has recovered (if r != 0)
        b_4_last = b_4 
        
        # This can also be used to determine if a fixed point is reached
        b_2_last = b_2
    
        # The fraction of total number of infections
        a_3 = np.count_nonzero(b_2) / float(n)
        
        # determine if the overall faction of infection exceed the threshold
        l_3 = -(t_3 - a_3) # Note that I cannot do a_3 - t_3 since a_3 is not a vector
        l_3[l_3 >= 0.0] = 1.0
        l_3[l_3 < 0.0] = 0.0
        #l3 = t_3 <= a_3
        
        # Determine if the fraction of neighbors with wear face masks exceeds a threshold
        l_2 = F @ b_1_last - t_2 # sparse? 
        l_2[l_2 >= 0.0] = 1.0
        l_2[l_2 < 0.0] = 0.0
        #l_2 = (F @ b_1_last) >= t_2 WORTH TRYING!
        
        # Update the mask state b_1
        b_1 = a_1 + l_2 + l_3 # logical operation?
        b_1[b_1 >= 1.0] = 1.0 # sparse?
        #b_1 = np.logical_or(np.logical_or(a_1, l_2), l_3) WORTH TRYING

        # The # of infected neighbors of each v
        d = A @ b_2_last
        
        # The # of infected neighbors with mask
        d_2 = A @ np.multiply(b_1_last, b_2_last) # Very important to pass b_1_last here

        # The # of infected neighbors without mask
        d_1 = d - d_2
        
        #Only susceptibles (b_4) can be infected 
        #--------------------------------------------------#
        # h1 : the probability of not getting infected from neighbors who do not wear masks (1 - p or 1 - alpha p)
        temp = one - (b_1 * (1.0 - alpha)) # IMPORTANT!!!!! b_1_last? (syn vs asyn)
        h_1 = one - (temp * p)
            
        # h2: contains the probability of not getting infected from neighbors who wear masks (1 - beta p or 1 - alpha beta p)
        h_2 = one - (temp * beta * p)

        temp = np.multiply(np.power(h_1, d_1), np.power(h_2, d_2))
        q = np.multiply(b_4, one - temp)
        #--------------------------------------------------#
        
        # Has to flatten q to pass it to the binomial funciton
        q_f = q.flatten()
        
        # Compute newly infected nodes
        newly_infected = np.reshape(np.random.binomial(1, q_f), (-1 ,1))
        
        # Recovery
        rr = np.random.choice([0, 1], size = (n, 1), p=[1.0 - r, r])
        b_2 = b_2 - rr
        b_2[b_2 == -1] = 0.0
        
        # Update b_2
        b_2 = newly_infected + b_2
        
        # Update the susceptible vector
        b_4 = -b_2 - b_3
        b_4[b_4 == 0.0] = 1.0
        b_4[b_4 < 0.0] = 0.0
        
        # Determine if a fixed point it reached
        if np.array_equal(b_2, b_2_last):
            return np.count_nonzero(b_2)

    return np.count_nonzero(b_2)