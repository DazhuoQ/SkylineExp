import torch
from torch_geometric.utils import k_hop_subgraph
from typing import List, Dict
import numpy as np
# from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

from src.utils import *

def get_l_hop_neighbors(node_idx: int, l: int, edge_index: torch.Tensor) -> set:
    subset, _, _, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=l,
        edge_index=edge_index,
        relabel_nodes=False
    )
    return set(subset.tolist()) - {node_idx}

def build_jaccard_similarity_matrix(test_nodes: List[int], edge_index: torch.Tensor, l: int):
    neighborhoods = [get_l_hop_neighbors(node, l, edge_index) for node in test_nodes]
    n = len(test_nodes)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            inter = len(neighborhoods[i] & neighborhoods[j])
            union = len(neighborhoods[i] | neighborhoods[j])
            sim = inter / union if union > 0 else 0.0
            sim_matrix[i][j] = sim_matrix[j][i] = sim
    return sim_matrix

def balanced_k_medoids_clustering(sim_matrix: np.ndarray, m: int, max_size_diff: int = 1):
    """
    Clusters using K-Medoids with balanced constraints (approximate).
    """
    n = sim_matrix.shape[0]
    kmedoids = KMedoids(n_clusters=m, metric="precomputed", init='k-medoids++', random_state=42)
    kmedoids.fit(1 - sim_matrix)  # because distance = 1 - similarity
    labels = kmedoids.labels_

    # Balance clusters if necessary
    from collections import defaultdict
    from random import shuffle

    cluster_map = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_map[label].append(idx)

    # Flatten and rebalance
    all_nodes = [node for cluster in cluster_map.values() for node in cluster]
    shuffle(all_nodes)
    balanced_clusters = [all_nodes[i::m] for i in range(m)]
    return balanced_clusters

# Usage
def cluster_test_nodes(test_nodes: List[int], edge_index: torch.Tensor, l: int, m: int):
    sim_matrix = build_jaccard_similarity_matrix(test_nodes, edge_index, l)
    balanced_clusters = balanced_k_medoids_clustering(sim_matrix, m)
    return [torch.tensor([test_nodes[i] for i in cluster]) for cluster in balanced_clusters]


# config = load_config("config.yaml")
# L = config['L']
# # VT = config['VT']
# m = config['m']
# random_seed = config['random_seed']
# set_seed(random_seed)
# data = dataset_func(config)

# VT = torch.load('./results/test_nodes.pt')

# clusters = cluster_test_nodes(VT, data.edge_index, l=L, m=m)
# print(clusters)
# torch.save(clusters, './results/partition.pt')
