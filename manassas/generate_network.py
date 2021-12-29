import networkx as nx
import numpy as np
from scipy import sparse

G_1 = nx.read_edgelist("manassas_social.edges")
G_2 = nx.read_edgelist("manassas_bio.edges")

print("Social layer: ")
print(nx.info(G_1))

print("Bio layer: ")
print(nx.info(G_2))

path_1 = "manassas_social.npz"
path_2 = "manassas_bio.npz"

A_1 = nx.to_scipy_sparse_matrix(G_1)
A_2 = nx.to_scipy_sparse_matrix(G_2)

sparse.save_npz(path_1, A_1)
sparse.save_npz(path_2, A_2)

