import networkx as nx
import sys
import numpy as np
from scipy import sparse
import numpy as np
from scipy.sparse import diags

n = 30000 # The number of vertices
avg_deg = int(sys.argv[1])
m = int(avg_deg / 2) # The number of edges added to each new node

G = nx.generators.random_graphs.barabasi_albert_graph(n, m)
#G = nx.generators.random_graphs.powerlaw_cluster_graph(n, m, 5/6)
G = nx.convert_node_labels_to_integers(G)

print(nx.info(G))

# Construct G_1 and G_2
G_1 = G
G_2 = G

p =  0.004

additional_edges = int(G.number_of_edges() * p)
while additional_edges != 0:
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    if i != j and (not G_1.has_edge(i, j)):
        G_1.add_edge(i,j)
        additional_edges -= 1

additional_edges = int(G.number_of_edges() * p)
while additional_edges != 0:
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    if i != j and (not G_2.has_edge(i, j)):
        G_2.add_edge(i,j)
        additional_edges -= 1


path_1 = "ba_30000_" + str(avg_deg) + "_1.npz"
path_2 = "ba_30000_" + str(avg_deg) + "_2.npz"

A_1 = nx.to_scipy_sparse_matrix(G_1)
A_2 = nx.to_scipy_sparse_matrix(G_2)

sparse.save_npz(path_1, A_1)
sparse.save_npz(path_2, A_2)

print(nx.info(G_1))
print(nx.info(G_2))
print("-----------------------------")
