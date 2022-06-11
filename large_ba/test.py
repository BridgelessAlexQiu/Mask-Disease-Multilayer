from scipy import sparse
import powerlaw
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

file_name= "ba_30000_10_1.npz"

A = sparse.load_npz(file_name)

G = nx.convert_matrix.from_scipy_sparse_matrix(A)

degree_sequence = [d for n, d in G.degree()]  # degree sequence


#results = powerlaw.Fit(degree_sequence)

#print(results.power_law.alpha)

print(nx.info(G))
