import mms.threshold as mt
import math
import matplotlib.pyplot as plt
from guppy import hpy
from scipy import sparse
import sys
import time
import numpy as np
import sys
from scipy.sparse import diags

if __name__ == "__main__":
    trans_prob = 0
    damping_factor = float(sys.argv[1])

    final_size_vector = []
    max_frac_vector = []
    days_vector = []
    mask_number_vector = []

    # file_name = "results_part_1/similar/mask_effectiveness/p_mask_effectiveness_" + str(damping_factor) + ".txt"
    file_name = "results_part_1/similar/mask_effectiveness/p_mask_effectiveness_" + str(damping_factor) + ".txt"

    while trans_prob < 1:
        trans_prob += 0.01
        if trans_prob >= 1.0: #avoid overflow
            trans_prob = 1.0
        print("Transmission Probability: {}".format(trans_prob))
    
        # Number of contagions
        c = 2

        # Number of maximum time-steps: to compute R0, set it to 2
        num_iter = 1000

        # percentage of initial infection *  Start with 10 infected individual
        infection_dens = 0.0003

        # percentage of initial mask wearing * 
        mask_prob = 0

        # percentage of prosocial
        pro_soc_p = 0.01

        # Fear of the disease ranges from 
        low_fear = 0.001
        high_fear = 0.15

        # Peer pressure
        low_peer = 0.3
        high_peer = 1
 
        # damping factors * TO COMPUTE R0, WE SET MASK TO BE USELSSS
        alpha = damping_factor  # if the susceptible wears a mask
        beta = damping_factor # if the infected wears a mask

        # recovery rate * 
        r = float(1/9)

        # covid_mask(A_1, A_2, D_inverse, a_1, t_2, t_3, b_1, b_2, p, alpha, beta, r, k)
        f = mt.covid_mask
    
        # Netowrk path name:
        path_1 = "large_ba/ba_30000_10_1.npz"
        path_2 = "large_ba/ba_30000_10_2.npz"

        # The adjacency matrix
        A_1 = sparse.load_npz(path_1) # social layer
        A_2 = sparse.load_npz(path_2) # bio layer

        # Numbe of nodes
        n = int(np.shape(A_1)[0])

        # Number of edges
        m_1 = A_1.count_nonzero()/2
        m_2 = A_2.count_nonzero()/2

        # The inverse degree matrix
        d = np.reciprocal((A_1 @ np.ones((n, 1), dtype = 'float')).flatten())
        D_inverse = diags(d, shape=(n, n))
        D_inverse = sparse.csr_matrix(D_inverse) #Convert D_inverse to a sparse matrix
    
        average_final_size = 0
        average_days = 0
        average_max_fraction = 0
        average_mask_number = 0
        itr = 50
        for _ in range(itr):
            # The prosocial vector a_1
            a_1 = sparse.random(n, 1, density = pro_soc_p, format='csr', data_rvs=np.ones)

            # The vector of the thresholds of a_2  (the faction of neighbors that wear masks) *
            t_2 = np.random.uniform(low = low_peer, high = high_peer, size = (n, 1))

            # The vector of the thresholds of a_3  (the faction of overall infection to start wearing masks) *
            t_3 = np.random.uniform(low = low_fear, high = high_fear, size = (n, 1))

            # The inital mask wearing
            b_1 = np.random.choice([0, 1], size = (n, 1), p=[1.0 - mask_prob, mask_prob])

            # The initial infection vector
            b_2 = np.random.choice([0, 1], size = (n, 1), p=[1.0 - infection_dens, infection_dens])
        
            # The starting processing time
            start = time.time()
            
            final_size, max_frac, days, mask_number = f(A_1, A_2, D_inverse, a_1, t_2, t_3, b_1, b_2, trans_prob, alpha, beta, r, num_iter)

            average_final_size += final_size
            average_days += days
            average_max_fraction += max_frac
            average_mask_number += mask_number

            # The ending processing time
            end = time.time()

        final_size_vector.append(average_final_size / itr)
        max_frac_vector.append(average_max_fraction / itr)
        days_vector.append(average_days / itr)
        mask_number_vector.append(average_mask_number / itr)
   
    f = open(file_name, 'w')

    f.write("Attack rate vector:\n")
    f.write(str(final_size_vector))
    f.write('\n\n')

    f.write("Duration vector:\n")
    f.write(str(days_vector))
    f.write('\n\n')

    f.write("Mask acceptance rate vector:\n")
    f.write(str(mask_number_vector))
    f.write('\n\n')

    f.close()
