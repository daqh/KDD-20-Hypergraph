
import networkx as nx
import hypernetx as hnx
from networkx.algorithms import bipartite
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import torch
from deep_hyperlink_prediction.models.node2vec_slp import Node2VecSLP
from deep_hyperlink_prediction.utils import datasets, load_dataset
import logging
from datetime import datetime
import itertools

def hyper_pa(S: list[int], NP: list[int], n: int, trials: int, alpha: float, model: torch.nn.Module):
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
    s_max = len(S)
    for i in range((s_max//2)):
        G.add_node(i + 1, bipartite=1)
        G.add_node(s_max - i - 1, bipartite=1)
        G.add_node(f'a{i + 1}', bipartite=0)
        G.add_edges_from([(f'a{i + 1}', s_max - i - 1)])
        G.add_edges_from([(f'a{i + 1}', i + 1)])

    for i in  range(1, n + 1):
        logging.info(f'{i}/{n}')
        k = np.random.choice(a = len(NP), size = 1, replace=False, p = [x/sum(NP) for x in NP])[0] + 1 # Sample a number k from NP
        logging.debug(f'\tk = {k}')
        for j in range(1, k + 1):
            s = np.random.choice(a = len(S), size = 1, replace=False, p = [x/sum(S) for x in S])[0] + 1 # Sample a hyperedge size s from S
            logging.debug(f'\ts = {s}')
            if s == 1:
                logging.debug(f'\tAdding hyperedge {i}')
                G.add_node(i, bipartite=1)
                G.add_node(f'c{i}', bipartite=0)
                G.add_edge(i, f'c{i}') # Add the hyperedge {i} to G
            # else if all (s-1)-sized groups have 0 degree then
            elif len({n for n, d in G.nodes(data=True) if d["bipartite"] == 1 and d != i}) < s - 1: # TODO: verifica correttezza
                # ----------------- #
                # for t in range(trials):
                # #     # Choose s-1 nodes randomly
                # nodes = random.choices(range(1, n), k=s-1)
                #     print(torch.tensor(nodes))
                # ----------------- #
                # Choose s-1 nodes randomly
                logging.debug(f'\tChoosing {s - 1} nodes randomly')
                nodes = random.choices(range(1, n), k=s-1)
                nodes.append(i)
                # Add the hyperedge of i and the s-1 nodes to G
                G.add_nodes_from(nodes, bipartite=1)
                G.add_node(f'c{i}-{j}', bipartite=0)
                G.add_edges_from([(l, f'c{i}-{j}') for l in nodes])
            else:
                # ----------------- #
                # top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1 and d != i}
                # if i in top_nodes:
                #     top_nodes.remove(i)
                # should_break = False
                # if s > 0:
                #     for group in itertools.combinations(top_nodes, s - 1):
                #         group = list(group)
                #         group.append(i)
                #         group = torch.tensor(group)
                #         p = model(group)
                #         if p >= alpha:
                #             should_break = True
                #             G.add_nodes_from(top_nodes, bipartite=1)
                #             G.add_node(i, bipartite=1)
                #             G.add_node(f'c{j}', bipartite=0)
                #             G.add_edge(i, f'c{j}')
                #             G.add_edges_from([(l, f'c{j}') for l in top_nodes])
                #             break
                # if should_break:
                #     continue
                # ----------------- #
                # Choose a group of size s-1 with probability proportional to degree
                logging.debug(f'\tChoosing a group of size {s - 1} with probability proportional to degree')
                top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1 and d != i}
                nodes = random.choices(list(top_nodes), weights=[G.degree[i] for i in top_nodes], k=s-1)
                nodes.append(i)
                # Add the hyperedge of i and the s-1 nodes to G
                G.add_nodes_from(nodes, bipartite=1)
                G.add_node(f'c{i}-{j}', bipartite=0)
                G.add_edges_from([(l, f'c{i}-{j}') for l in nodes])
    return G

def main(args):
    logging.basicConfig(level=logging.INFO)

    nodes = args.nodes
    trials = args.trials
    alpha = args.alpha
    dataset = args.dataset

    begin = datetime.now()

    with open(f"simplex per node/{dataset}-simplices-per-node-distribution.txt") as f:
        NP = [int(x) for x in f.read().splitlines()]

    with open(f"size distribution/{dataset} size distribution.txt") as f:
        S = [int(x) for x in f.read().splitlines()]

    _, edge_index, _ = load_dataset(datasets['email-Eu'])

    model = Node2VecSLP(edge_index, 256, 1, aggregate="sum")
    model.load_state_dict(torch.load('deep_hyperlink_prediction/pretrained_models/node2vec_slp-email-Eu-sum.pt'))

    H = hyper_pa(S, NP, nodes, 20, 0.5, model)
    H = hnx.Hypergraph.from_bipartite(H)

    end = datetime.now()

    logging.info(f'Elapsed time: {end - begin}')
    logging.info(f'Number of nodes: {H.number_of_nodes()}')

    logging.debug('Writing output file')
    
    with open('output_directory/output.txt', 'w') as f:
        for e in H.edges:
            nodes = sorted(H.edges[e])
            f.write(str(' ').join(map(str, nodes)) + '\n')
    
    logging.debug('Writing output.unique file')

    hyperedges = []
    with open('output_directory/output.unique.txt', 'w') as f:
        for edge in H.edges:
            nodes = sorted(H.edges[edge])
            r = str(' ').join(map(str, nodes))
            if r not in hyperedges:
                hyperedges.append(r)
                f.write(r + '\n')

    # Print the list of nodes in each hyperedge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hypergraph Preferential Attachment')
    parser.add_argument('--nodes', '-n', dest='nodes', type=int, default=200, help='Number of nodes')
    parser.add_argument('--dataset', '-d', dest='dataset', type=str, default='email-Eu', help='Dataset')
    parser.add_argument('--alpha', '-a', dest='alpha', type=float, default=1, help='Awareness parameter')
    parser.add_argument('--trials', '-t', dest='trials', type=int, default=20, help='Number of trials')
    args = parser.parse_args()
    main(args)
