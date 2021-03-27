# %%
import networkx as nx
import NMFA
from time import time
import netrd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigvals

def node_degree_xyl(G):
    nodes = set(G)
    xdeg = G.degree
    ydeg = G.degree

    for u, degu in xdeg(nodes):
        neighbors = (nbr for _, nbr in G.edges(u) if nbr in nodes)
        for v, degv in ydeg(neighbors):
            yield degu, degv, nx.shortest_path_length(G, u, v)

def correlation_distance(G1, G2):
    list_g1 = list(node_degree_xyl(G1))
    list_g2 = list(node_degree_xyl(G2))
    max_list1 = np.max(list_g1,axis=0)
    max_list2 = np.max(list_g2,axis=0)
    max_x = np.max([max_list1[0], max_list2[0]])
    max_y = np.max([max_list1[1], max_list2[1]])
    max_l = np.max([max_list1[2], max_list2[2]])
    p1 = np.zeros([max_x+1, max_y+1, max_l+1])
    p2 = np.zeros([max_x+1, max_y+1, max_l+1])

    for u,v,l in list_g1:
        p1[u,v,l] += 1
    p1 /= len(list_g1)

    for u,v,l in list_g2:
        p2[u,v,l] += 1
    p2 /= len(list_g2)

    return 0.5 * np.sum(np.abs(p1 - p2))

def spectral_distance(G1, G2):
    adj1 = nx.to_numpy_array(G1)
    adj2 = nx.to_numpy_array(G2)
    L1 = laplacian(adj1, normed=False)
    L2 = laplacian(adj2, normed=False)
    v1 = np.sort(eigvals(L1))
    v2 = np.sort(eigvals(L2))
    u1 = np.sort(eigvals(adj1))
    u2 = np.sort(eigvals(adj2))
    return np.abs(np.sqrt(np.sum(np.square(u1-u2)+np.square(v1-v2))))

# %% Init
repeat_n = 10
G1 = []
G2 = []
for i in range(repeat_n):
    G1.append(nx.erdos_renyi_graph(50,0.8))
    G2.append(nx.erdos_renyi_graph(50,0.8))

# %% edit distance
start_time = time()
for i in range(repeat_n):
    print(i)
    nx.graph_edit_distance(G1[i], G2[i])
end_time = time()
print((end_time-start_time)/repeat_n)
# %% spectral distance
start_time = time()
dist = netrd.distance.IpsenMikhailov()
for i in range(repeat_n):
    print(i)
    dist.dist(G1[i], G2[i])
end_time = time()
print((end_time-start_time)/repeat_n)
# %% correlation distance
start_time = time()
for i in range(repeat_n):
    print(i)
    correlation_distance(G1[i], G2[i])
end_time = time()
print((end_time-start_time)/repeat_n)
# %% structure distance
start_time = time()
for i in range(repeat_n):
    ntauls_list = []
    ntauls_list.append(NMFA.nfd_nk(G1[i]))
    ntauls_list.append(NMFA.nfd_nk(G2[i]))
    ndim_list = []
    for i in range(len(ntauls_list)):
        dim, _ = NMFA.ndimension(ntauls_list[i])
        ndim_list.append(dim)
    dist = NMFA.distance(ndim_list)
end_time = time()
print((end_time-start_time)/repeat_n)