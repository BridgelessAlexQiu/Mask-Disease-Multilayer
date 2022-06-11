"""
Author: Zirou Qiu
Last modfied: 06/11/2022
Description: 
    This module consists of simulations of the spread of dueling 
    contigions
"""


#--------------------------- Imports ------------------------------#
import numpy as np
import mms.utility as mu
from scipy import sparse

def covid_mask(A_1, A_2, D_inverse, a_1, t_2, t_3, b_1, b_2, p, alpha, beta, r, k):
    """
        Description
        -----------
        This funciton simulates the spread of two contagions on a two-layer network.
        The function outputs (i) attack rate, (ii) duration, (iii) mask acceptance rate 
        (iv) maximum fraction of infection.
        
        Parameters
        ----------
        A_1: n x n scipy sparse matrix, int {0, 1}
           The adjacency matrix of the social layer

        A_2: n x n scipy sparse matrix, int {0, 1}
            The adjacency matrix of the disease layer
        
        D_inverse: n x n scipy sparse matrix, float [0, 1]
            The inversed diagonal matrix of the social layer.
        
        a_1: n x 1 scipy sparse matrix, int {0, 1}
            (a_1)_i = 1 if the person i is prosocial, and (a_1)_i = 0 otherwise.

        t_2: n x 1 numpy array, float [0, 1]
            (t_2)_i is threshold percentage of neighbors who wear masks for person 
            i to wear a mask in the next iteration.

        t_3: n x 1 numpy array, float [0, 1]
           (t_3)_i is the threshold percentage of the overall infection of the population
           for person i to wear a mask in the next iteration.

        b_1: n x 1 scipy sparse matrix, int {0, 1}
            (b_1)_i = 1 if the person i wears a mask at the current iteration.

        b_2: n x 1 scipy sparse matrix, int {0, 1}
            (b_2)_1 = 1 if the person i is infected by the disease at the current iteration
    
        p: float [0, 1]
            Transimission probability of the disease

        alpha: The damping factor on p when the person himself wears a mask.

        beta: The damping factor on p when a neighbor of a person wears a mask.

        r: Recovery probability.

        k: The maximum number of time-steps.
            
    """
    # Keep track of the dynamic: {time: [# of masks, # of infections]}
    # dynamic = {}
    
    # Compute the degree fraction matrix
    F = D_inverse @ A_1
    F = sparse.csr_matrix(F)

    # The number of vertices
    n = np.shape(A_1)[0]
    
    # The one and zero vectors
    one = np.ones((n, 1), dtype = 'float')
    zero = np.zeros((n, 1), dtype = 'float')

    # The recovery vector: b_3
    b_3 = np.zeros((n, 1), dtype = 'float') # Initially, no one has recovered. 

    # The susceptible vector: b_4
    b_4 = -b_2 - b_3
    b_4[b_4 == 0.0] = 1.0
    b_4[b_4 < 0.0] = 0.0

    # The largest fraction of infection reached throughout the time
    max_frac = 0.0

    # The number of days the infection lasts
    days = 0

    # total number of mask wearings (sum over all days)
    total_mask = 0

    # mask vector thoughout the time
    mask_vector = [] #new

    # infection vector
    infection_vector = [] #new

    # The main loop
    for i in range(k):
        days += 1 

        mask_vector.append(float(np.count_nonzero(b_1) / n)) #new

        infection_vector.append(float(np.count_nonzero(b_2) / n)) #new
 
        # dynamic[i] = [np.count_nonzero(b_1), np.count_nonzero(b_2), np.count_nonzero(b_3), np.count_nonzero(b_4)]

        # need b_1_last to update the state of the second contagion
        b_1_last = b_1 
        
        b_4_last = b_4 
    
        b_2_last = b_2
    
        # The fraction of total number of infections
        a_3 = np.count_nonzero(b_2) / float(n)
   
        # Update the max_frac
        if a_3 > max_frac:
            max_frac = a_3
        
        # determine if the overall faction of infection exceed the threshold
        l_3 = -(t_3 - a_3) # Note that I cannot do a_3 - t_3 since a_3 is not a vector
        l_3[l_3 >= 0.0] = 1.0
        l_3[l_3 < 0.0] = 0.0
        # l3 = t_3 <= a_3
        
        # Determine if the fraction of neighbors with wear face masks exceeds a threshold
        l_2 = F @ b_1_last - t_2 # sparse? 
        l_2[l_2 >= 0.0] = 1.0
        l_2[l_2 < 0.0] = 0.0
        # l_2 = (F @ b_1_last) >= t_2 WORTH TRYING!
        
        # Update the mask state b_1
        b_1 = a_1 + l_2 + l_3 # logical operation?
        b_1[b_1 >= 1.0] = 1.0 # sparse?
        #b_1 = np.logical_or(np.logical_or(a_1, l_2), l_3) WORTH TRYING

        total_mask += np.count_nonzero(b_1)

        # The # of infected neighbors of each v
        d = A_2 @ b_2_last
        
        # The # of infected neighbors with mask
        d_2 = A_2 @ np.multiply(b_1_last, b_2_last) # Very important to pass b_1_last here

        # The # of infected neighbors without mask
        d_1 = d - d_2
        
        # Only susceptibles (b_4) can be infected 
        #--------------------------------------------------#
        # h1 : the probability of not getting infected from neighbors who do not wear masks (1 - p or 1 - alpha p)
        temp = one - (b_1 * (1.0 - alpha)) # IMPORTANT: b_1_last vs b_1 (syn vs asyn)
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

        # Computer R0 (do this before recovery)
        # R_0 = np.count_nonzero(newly_infected) / np.count_nonzero(b_2)

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

        # A fixed point is reached under zero infection
        if np.array_equal(b_2, zero):
            # print("A fixed point is reached at iteration {}".format(i))
            average_mask = float(total_mask / days)
            return round(float(np.count_nonzero(b_3) / n), 4), round(max_frac, 4), days, round(float(average_mask / n), 4)

    average_mask = float(total_mask / days)
    return round(float(np.count_nonzero(b_3) / n), 4), round(max_frac, 4), days, round(float(average_mask / n), 4)

def covid_mask_time_series(A_1, A_2, D_inverse, a_1, t_2, t_3, b_1, b_2, p, alpha, beta, r, k):
    """
        Description
        -----------
        This funciton simulate the spread of two contagions on a two-layer network.
        The function outputs (i) fraction of infection per day
        (ii) fraction of mask-wearing per day.
        
        Parameters
        ----------
        A_1: n x n scipy sparse matrix, int {0, 1}
           The adjacency matrix of the social layer

        A_2: n x n scipy sparse matrix, int {0, 1}
            The adjacency matrix of the disease layer
        
        D_inverse: n x n scipy sparse matrix, float [0, 1]
            The inversed diagonal matrix of the social layer.
        
        a_1: n x 1 scipy sparse matrix, int {0, 1}
            (a_1)_i = 1 if the person i is prosocial, and (a_1)_i = 0 otherwise.

        t_2: n x 1 numpy array, float [0, 1]
            (t_2)_i is threshold percentage of neighbors who wear masks for person 
            i to wear a mask in the next iteration.

        t_3: n x 1 numpy array, float [0, 1]
           (t_3)_i is the threshold percentage of the overall infection of the population
           for person i to wear a mask in the next iteration.

        b_1: n x 1 scipy sparse matrix, int {0, 1}
            (b_1)_i = 1 if the person i wears a mask at the current iteration.

        b_2: n x 1 scipy sparse matrix, int {0, 1}
            (b_2)_1 = 1 if the person i is infected by the disease at the current iteration
    
        p: float [0, 1]
            Transimission probability of the disease

        alpha: The damping factor on p when the person himself wears a mask.

        beta: The damping factor on p when a neighbor of a person wears a mask.

        r: Recovery probability.

        k: The maximum number of time-steps.
            
    """
    # Keep track of the dynamic: {time: [# of masks, # of infections]}
    # dynamic = {}
    
    # Compute the degree fraction matrix
    F = D_inverse @ A_1
    F = sparse.csr_matrix(F)

    # The number of vertices
    n = np.shape(A_1)[0]
    
    # The one and zero vectors
    one = np.ones((n, 1), dtype = 'float')
    zero = np.zeros((n, 1), dtype = 'float')

    # The recovery vector: b_3
    b_3 = np.zeros((n, 1), dtype = 'float') # Initially, no one has recovered. 

    # The susceptible vector: b_4
    b_4 = -b_2 - b_3
    b_4[b_4 == 0.0] = 1.0
    b_4[b_4 < 0.0] = 0.0

    # The largest fraction of infection reached throughout the time
    max_frac = 0.0

    # The number of days the infection lasts
    days = 0

    # total number of mask wearings (sum over all days)
    total_mask = 0

    # mask vector thoughout the time
    mask_vector = [] #new

    # infection vector
    infection_vector = [] #new

    # The main loop
    for i in range(k):
        days += 1 

        mask_vector.append(float(np.count_nonzero(b_1) / n)) #new

        infection_vector.append(float(np.count_nonzero(b_2) / n)) #new
 
        # dynamic[i] = [np.count_nonzero(b_1), np.count_nonzero(b_2), np.count_nonzero(b_3), np.count_nonzero(b_4)]

        # need b_1_last to update the state of the second contagion
        b_1_last = b_1 
        
        b_4_last = b_4 
    
        b_2_last = b_2
    
        # The fraction of total number of infections
        a_3 = np.count_nonzero(b_2) / float(n)
   
        # Update the max_frac
        if a_3 > max_frac:
            max_frac = a_3
        
        # determine if the overall faction of infection exceed the threshold
        l_3 = -(t_3 - a_3) # Note that I cannot do a_3 - t_3 since a_3 is not a vector
        l_3[l_3 >= 0.0] = 1.0
        l_3[l_3 < 0.0] = 0.0
        # l3 = t_3 <= a_3
        
        # Determine if the fraction of neighbors with wear face masks exceeds a threshold
        l_2 = F @ b_1_last - t_2 # sparse? 
        l_2[l_2 >= 0.0] = 1.0
        l_2[l_2 < 0.0] = 0.0
        # l_2 = (F @ b_1_last) >= t_2 WORTH TRYING!
        
        # Update the mask state b_1
        b_1 = a_1 + l_2 + l_3 # logical operation?
        b_1[b_1 >= 1.0] = 1.0 # sparse?
        #b_1 = np.logical_or(np.logical_or(a_1, l_2), l_3) WORTH TRYING

        total_mask += np.count_nonzero(b_1)

        # The # of infected neighbors of each v
        d = A_2 @ b_2_last
        
        # The # of infected neighbors with mask
        d_2 = A_2 @ np.multiply(b_1_last, b_2_last) # Very important to pass b_1_last here

        # The # of infected neighbors without mask
        d_1 = d - d_2
        
        # Only susceptibles (b_4) can be infected 
        #--------------------------------------------------#
        # h1 : the probability of not getting infected from neighbors who do not wear masks (1 - p or 1 - alpha p)
        temp = one - (b_1 * (1.0 - alpha)) # IMPORTANT: b_1_last vs b_1 (syn vs asyn)
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

        # Computer R0 (do this before recovery)
        # R_0 = np.count_nonzero(newly_infected) / np.count_nonzero(b_2)

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

        # A fixed point is reached under zero infection
        if np.array_equal(b_2, zero):
            # print("A fixed point is reached at iteration {}".format(i))
            average_mask = float(total_mask / days)
            return infection_vector, mask_vector

    average_mask = float(total_mask / days)
    return infection_vector, mask_vector

