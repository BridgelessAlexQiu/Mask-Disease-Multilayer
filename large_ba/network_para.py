from scipy import sparse
import sys
import powerlaw
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

file_name= sys.argv[1]

A = sparse.load_npz(file_name)

G = nx.convert_matrix.from_scipy_sparse_matrix(A)

n = G.number_of_nodes()

degree_seq = [d for n, d in G.degree()]

d_min = min(degree_seq)
d_max = max(degree_seq)

results = powerlaw.Fit(degree_seq)

#L = nx.normalized_laplacian_matrix(G)

#e = np.linalg.eigvals(L.A)

#e = sort(e, reverse = True)

print("Power law exponnet: {}".format(results.power_law.alpha))

print("Average degree: {}".format(G.number_of_edges() * 2 / n))

print("Min degree: {}".format(d_min))

print("Max degree: {}".format(d_max))

print("Transitivity: {}".format(nx.transitivity(G)))

print("Average clustering coefficient: {}".format(nx.average_clustering(G)))

#print("Largest eigenvalue: ", max(e))

#print("Second eigenvalue: ", e[1])

