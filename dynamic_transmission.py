import mms.threshold as mt
import math
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import sys
import time
import numpy as np
import sys
from scipy.sparse import diags

if __name__ == "__main__":
    
    # The disease transmission probability
    # Initially set to 0, and increase by 0.01 to 1
    trans_prob = 0

    # The fraction of prosocial individuals
    pro_soc_p = 0.01
    
    # The averaged attack rate and std for each p
    attack_rate_vector = []
    attack_rate_std_vector = []

    # The highest fraction of infection and std for each p
    max_frac_vector = []
    max_frac_std_vector = []
    
    # The averaged duration and std for each p
    days_vector = []
    days_std_vector = []
    
    # The averaged mask acceptance rate and std for each p
    mask_number_vector = []
    mask_number_std_vector = []

    # The result is stored here
    file_name = "results/p_vs_final_size.txt"
    
    # Increase p from 0 to 1
    while trans_prob < 1:
        trans_prob += 0.01
        if trans_prob >= 1.0: # Avoid overflow
            trans_prob = 1.0
        print("Transmission Probability: {}".format(trans_prob))
    
        # Number of contagions
        c = 2

        # Number of maximum time-steps
        num_iter = 1000

        # percentage of initial infection. Start with 10 infected individual
        infection_dens = 0.0003

        # percentage of initial mask wearing
        mask_prob = 0

        # Fear of the disease ranges from 
        low_fear = 0.001
        high_fear = 0.15

        # Peer pressure
        low_peer = 0.3
        high_peer = 1
 
        # damping factors
        alpha = 0.3
        beta = 0.1 

        # recovery rate
        r = float(1/9)

        # covid_mask(A_1, A_2, D_inverse, a_1, t_2, t_3, b_1, b_2, p, alpha, beta, r, k)
        f = mt.covid_mask
        
        # Each data point is averaged over 100 cascades
        itr = 100

        # The vectors to store results
        itr_attack_rate = [None] * itr
        itr_days = [None] * itr
        itr_max_frac = [None] * itr
        itr_mask_numer = [None] * itr

        for ind in range(itr):
            n = 30000 # The number of vertices
            avg_deg = 10
            
            # Construct G
            m = int(avg_deg / 2)
            G = nx.generators.random_graphs.barabasi_albert_graph(n, m)
            G = nx.convert_node_labels_to_integers(G)

            p_prime =  0.004 # fraction of additional vertices to add

            # Construct G1
            G_1 = G
            additional_edges = int(G.number_of_edges() * p_prime)
            while additional_edges != 0:
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i != j and (not G_1.has_edge(i, j)):
                    G_1.add_edge(i,j)
                    additional_edges -= 1

            # Construct G2
            G_2 = G
            additional_edges = int(G.number_of_edges() * p_prime)
            while additional_edges != 0:
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i != j and (not G_2.has_edge(i, j)):
                    G_2.add_edge(i,j)
                    additional_edges -= 1
            
            # The adjacency matrix
            A_1 = nx.to_scipy_sparse_matrix(G_1) # social layer
            A_2 = nx.to_scipy_sparse_matrix(G_2) # bio layer

            # Numbe of nodes
            n = int(np.shape(A_1)[0])

            # Number of edges
            m_1 = A_1.count_nonzero()/2
            m_2 = A_2.count_nonzero()/2

            # The inverse degree matrix
            d = np.reciprocal((A_1 @ np.ones((n, 1), dtype = 'float')).flatten())
            D_inverse = diags(d, shape=(n, n))
            D_inverse = sparse.csr_matrix(D_inverse) # Convert D_inverse to a sparse matrix

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
            
            # Function call
            final_size, max_frac, days, mask_number = f(A_1, A_2, D_inverse, a_1, t_2, t_3, b_1, b_2, trans_prob, alpha, beta, r, num_iter)

            itr_attack_rate[ind] = final_size
            itr_max_frac[ind] = max_frac
            itr_days[ind] = days
            itr_mask_numer[ind] = mask_number

            # The ending processing time
            end = time.time()

        attack_rate_vector.append(np.mean(itr_attack_rate))
        attack_rate_std_vector.append(np.std(itr_attack_rate))

        max_frac_vector.append(np.mean(itr_max_frac))
        max_frac_std_vector.append(np.std(itr_max_frac))

        days_vector.append(np.mean(itr_days))
        days_std_vector.append(np.std(itr_days))

        mask_number_vector.append(np.mean(itr_mask_numer))
        mask_number_std_vector.append(np.std(itr_mask_numer))


    f = open(file_name, 'w')

    f.write("Attack rate vector:\n")
    f.write(str(attack_rate_vector))
    f.write('\n\n')

    f.write("Attack rate std vector:\n")
    f.write(str(attack_rate_std_vector))
    f.write('\n\n')

    f.write("Duration vector:\n")
    f.write(str(days_vector))
    f.write('\n\n')

    f.write("Duration std vector:\n")
    f.write(str(days_std_vector))
    f.write('\n\n')

    f.write("Mask acceptance rate vector:\n")
    f.write(str(mask_number_vector))
    f.write('\n\n')

    f.write("Mask acceptance rate std number vector:\n")
    f.write(str(mask_number_std_vector))
    f.write('\n\n')

    f.write("Max fraction vector:\n")
    f.write(str(max_frac_vector))
    f.write('\n\n')

    f.write("Max fraction std vector:\n")
    f.write(str(max_frac_std_vector))
    f.write('\n\n')

    f.close()
