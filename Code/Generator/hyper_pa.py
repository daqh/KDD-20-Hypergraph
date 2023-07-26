
import networkx as nx
import hypernetx as hnx
from networkx.algorithms import bipartite
import random
import math
import matplotlib.pyplot as plt
import argparse

def hyper_pa(S: list[int], NP: list[int], n: int, trials: int, alpha: float):
    '''Hypergraph Preferential Attachment
    
    Parameters
    ----------
    S : list[int]
        Distribution of hyperedge sizes (with max size s_max)
    NP : list[int]
        Distribution of number of new hyperedges
    n : int
        Number of nodes
    trials : int
        Number of trials
    alpha : float
        Awareness parameter
    '''
    G = nx.Graph()
    # Initialize G with s_max/2 disjoint hyperedges of size 2
    s_max = max(S)
    for i in range(s_max//2 + 1):
        G.add_node(i + 1, bipartite=1)
        G.add_node(s_max - i, bipartite=1)
        G.add_node(f'c{i + 1}', bipartite=0)
        G.add_edges_from([(f'c{i + 1}', s_max - i)])
        G.add_edges_from([(f'c{i + 1}', i + 1)])

    for i in  range(s_max//2 + 1, n + 1):
        print(i)
        k = random.choice(NP) # Sample a number k from NP
        for j  in range(1, k + 1):
            s = random.choice(S) # Sample a hyperedge size s from S
            if s == 1:
                G.add_node(i, bipartite=1)
                G.add_node(f"c{j}", bipartite=0)
                G.add_edge(i, f'c{i}') # Add the hyperedge {i} to G
            # else if all (s-1)-sized groups have 0 degree then
            elif len(G.nodes) < s - 1 or all(G.degree[i] == 0 for i in {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}):
                # Choose s-1 nodes randomly
                nodes = random.choices(range(1, n), k=s-1)
                # Add the hyperedge of i and the s-1 nodes to G
                G.add_nodes_from(nodes, bipartite=1)
                G.add_node(i, bipartite=1)
                G.add_node(f'c{j}', bipartite=0)
                G.add_edge(i, f'c{j}')
                G.add_edges_from([(l, f'c{j}') for l in nodes])
            else:
                # Choose a group of size s-1 with probability proportional to degree
                top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
                nodes = random.choices(list(top_nodes), weights=[G.degree[i] for i in top_nodes], k=s-1)
                # Add the hyperedge of i and the s-1 nodes to G
                G.add_nodes_from(nodes, bipartite=1)
                G.add_node(i, bipartite=1)
                G.add_node(f'c{j}', bipartite=0)
                G.add_edge(i, f'c{j}')
                G.add_edges_from([(l, f'c{j}') for l in nodes])
    return G

def main(*args, **kwargs):
    with open("simplex per node/email-Eu-simplices-per-node-distribution.txt") as f:
        S = [int(x) for x in f.read().splitlines()]
    with open("size distribution/email-Eu size distribution.txt") as f:
        NP = [int(x) for x in f.read().splitlines()]
    H = hyper_pa(S, NP, 300, 20, 1)
    H = hnx.Hypergraph.from_bipartite(H)
    with open('output_directory/output.txt', 'w') as f:
        for e in H.edges:
            nodes = sorted(H.edges[e])
            f.write(str(' ').join(map(str, nodes)) + '\n')
    # Print the nodes in the edge 1
    # print(H.incidence_dict)
    # hnx.draw(H)
    # plt.show()
    # Print the list of nodes in each hyperedge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypergraph Preferential Attachment')
    main()