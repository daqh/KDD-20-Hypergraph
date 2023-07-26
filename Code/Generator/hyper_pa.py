
import networkx as nx
import hypernetx as hnx
import random
import math
import matplotlib.pyplot as plt

def hyper_pa(S: list[int], NP: list[int], n: int):
    '''Hypergraph Preferential Attachment
    
    Parameters
    ----------
    S : list[int]
        Distribution of hyperedge sizes (with max size s_max)
    NP : list[int]
        Distribution of number of new hyperedges
    n : int
        Number of nodes
    '''
    G = nx.Graph()
    # Initialize G with s_max/2 disjoint hyperedges of size 2
    s_max = max(S)
    for i in range(s_max//2 + 1):
        G.add_node(i + 1, bipartite=0)
        G.add_node(s_max - i, bipartite=0)
        G.add_node(str(i + 1), bipartite=1)
        G.add_edges_from([(str(i + 1), s_max - i)])
        G.add_edges_from([(str(i + 1), i + 1)])

    for i in  range(1, n):
        k = random.choice(NP) # Sample a number k from NP
        for j  in range(1, k):
            s = random.choice(S) # Sample a hyperedge size s from S
            if s == 1:
                G.add_node(i, bipartite=0)
                G.add_node(str(i), bipartite=1)
                G.add_edges_from([(i, str(i))]) # Add the hyperedge {i} to G
            # else if all (s-1)-sized groups have 0 degree then
    return G

def main():
    H = hyper_pa([1, 15, 24], [1, 2, 3], 40)
    print(list(H.edges))
    H = hnx.Hypergraph.from_bipartite(H)
    # Print nodes in each hyperedge
    hnx.draw(H)
    plt.show()

if __name__ == '__main__':
    main()