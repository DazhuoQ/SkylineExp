from src.utils import *

import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from datasketch import MinHash, MinHashLSH
import random
import time

########################################
# 1. L-hop neighbor extraction
########################################

def get_l_hop_neighbors(node_idx, l, edge_index):
    subset, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=l, edge_index=edge_index, relabel_nodes=False)
    return set(subset.tolist()) - {node_idx}

def get_all_l_hop_sets(node_list, l, edge_index):
    l_hop_sets = {}
    for node in node_list:
        l_hop_sets[node] = get_l_hop_neighbors(node, l, edge_index)
    return l_hop_sets

########################################
# 2. Exact Similarity Clustering
########################################

def compute_similarity_matrix(nodes, l_hop_sets):
    n = len(nodes)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            set_i = l_hop_sets[nodes[i]]
            set_j = l_hop_sets[nodes[j]]
            if not set_i or not set_j:
                sim = 0.0
            else:
                sim = len(set_i & set_j) / len(set_i | set_j)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # symmetric
    return sim_matrix


# def cluster_nodes_exact(similarity_matrix, num_clusters):
#     distance_matrix = 1 - similarity_matrix
#     clustering = AgglomerativeClustering(
#         n_clusters=num_clusters,
#         metric='precomputed',
#         linkage='average'
#     )
#     labels = clustering.fit_predict(distance_matrix)
#     return labels


def cluster_nodes_exact_safe(test_nodes, l_hop_sets, num_clusters):
    sim_matrix = compute_similarity_matrix(test_nodes, l_hop_sets)
    distance_matrix = 1 - sim_matrix

    # Make sure distance matrix is valid
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)

    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)

    assigned_nodes = set()
    clusters = [[] for _ in range(num_clusters)]

    for node, label in zip(test_nodes, labels):
        clusters[label].append(node)
        assigned_nodes.add(node)

    # Step 2: Handle missing nodes
    missing_nodes = set(test_nodes) - assigned_nodes
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} missing nodes detected. Assigning them randomly to clusters.")
        for idx, node in enumerate(missing_nodes):
            clusters[idx % num_clusters].append(node)

    return clusters



def structure_preserving_balance(raw_clusters, target_num_clusters):
    """
    Balance clusters to make them similar in size without random shuffling.
    Minimal disturbance: split large clusters, merge small ones.
    """
    total_nodes = sum(len(c) for c in raw_clusters)
    target_size = total_nodes // target_num_clusters
    min_size = int(0.9 * target_size)
    max_size = int(1.1 * target_size)

    # Step 1: Split very large clusters
    refined_clusters = []
    for cluster in raw_clusters:
        if len(cluster) <= max_size:
            refined_clusters.append(cluster)
        else:
            # split large cluster into chunks
            chunks = [cluster[i:i + target_size] for i in range(0, len(cluster), target_size)]
            refined_clusters.extend(chunks)

    # Step 2: Merge very small clusters
    small_clusters = [c for c in refined_clusters if len(c) < min_size]
    normal_clusters = [c for c in refined_clusters if len(c) >= min_size]

    # Greedy merging
    merged_clusters = []
    current = []
    for cluster in small_clusters:
        current.extend(cluster)
        if len(current) >= min_size:
            merged_clusters.append(current)
            current = []
    if current:
        merged_clusters.append(current)

    # Step 3: Final clusters
    final_clusters = normal_clusters + merged_clusters

    # Step 4: If still not exactly target_num_clusters, slight adjustment
    while len(final_clusters) > target_num_clusters:
        # Merge two smallest clusters
        final_clusters = sorted(final_clusters, key=lambda x: len(x))
        merged = final_clusters[0] + final_clusters[1]
        final_clusters = [merged] + final_clusters[2:]
    while len(final_clusters) < target_num_clusters:
        # Split largest cluster
        final_clusters = sorted(final_clusters, key=lambda x: -len(x))
        largest = final_clusters.pop(0)
        mid = len(largest) // 2
        final_clusters.append(largest[:mid])
        final_clusters.append(largest[mid:])

    return final_clusters



########################################
# 3. MinHash + LSH Approximate Clustering
########################################

def build_minhash(neighbor_set, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for elem in neighbor_set:
        m.update(str(elem).encode('utf8'))
    return m

# def lsh_clustering(test_nodes, l_hop_sets, threshold=0.5, num_perm=128):
#     test_nodes = list(set(test_nodes))

#     lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
#     node2minhash = {}

#     for node in test_nodes:
#         m = build_minhash(l_hop_sets[node], num_perm=num_perm)
#         node2minhash[node] = m
#         lsh.insert(str(node), m)

#     cluster_buckets = {}
#     visited = set()

#     for node in test_nodes:
#         if node in visited:
#             continue
#         similar_nodes = lsh.query(node2minhash[node])
#         similar_nodes = [int(x) for x in similar_nodes]
#         for n in similar_nodes:
#             visited.add(n)
#         cluster_buckets[node] = similar_nodes

#     buckets = list(cluster_buckets.values())
#     return buckets
def lsh_clustering_unique(test_nodes, l_hop_sets, threshold=0.5, num_perm=128):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    node2minhash = {}
    inserted = set()

    for node in test_nodes:
        if node in inserted:
            continue
        m = build_minhash(l_hop_sets[node], num_perm=num_perm)
        node2minhash[node] = m
        try:
            lsh.insert(str(node), m)
            inserted.add(node)
        except ValueError:
            # If somehow already inserted, skip
            continue

    cluster_buckets = []
    assigned = set()

    for node in test_nodes:
        if node in assigned:
            continue
        similar_nodes = lsh.query(node2minhash[node])
        unique_group = [int(n) for n in similar_nodes if int(n) not in assigned]
        if unique_group:
            cluster_buckets.append(unique_group)
            assigned.update(unique_group)

    return cluster_buckets





from collections import deque
import heapq
import numpy as np
from itertools import combinations

def jaccard_sim(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

def split_large_bucket(bucket, l_hop_sets, target_size):
    """
    Greedily split a large bucket into chunks of roughly target_size
    using similarity-based seed growing.
    """
    bucket = list(bucket)
    used = set()
    clusters = []

    while len(used) < len(bucket):
        # pick a seed not used
        seed = next(n for n in bucket if n not in used)
        cluster = [seed]
        used.add(seed)

        # score candidates based on similarity to current cluster
        candidates = [(jaccard_sim(l_hop_sets[seed], l_hop_sets[n]), n)
                      for n in bucket if n not in used]
        candidates.sort(reverse=True)

        for sim, n in candidates:
            if len(cluster) >= target_size:
                break
            cluster.append(n)
            used.add(n)

        clusters.append(cluster)
    return clusters

def greedy_merge_small_buckets(small_buckets, m, target_size):
    """
    Merge small buckets into m clusters using a greedy size-aware strategy.
    """
    clusters = [[] for _ in range(m)]
    bucket_queue = deque(sorted(small_buckets, key=lambda b: -len(b)))

    while bucket_queue:
        bucket = bucket_queue.popleft()
        # find cluster with most space
        sizes = [len(c) for c in clusters]
        idx = sizes.index(min(sizes))
        clusters[idx].extend(bucket)

    return clusters

def refined_balance_lsh_clusters(buckets, target_num_clusters, l_hop_sets, tolerance=0.1):
    """
    More refined LSH bucket balancing using structure-aware splitting and merging.
    """
    total_nodes = sum(len(b) for b in buckets)
    target_size = total_nodes // target_num_clusters
    max_size = int(target_size * (1 + tolerance))
    min_size = int(target_size * (1 - tolerance))

    large = []
    small = []

    for b in buckets:
        if len(b) > max_size:
            large.append(b)
        else:
            small.append(b)

    # Step 1: Split large buckets
    refined_chunks = []
    for b in large:
        splits = split_large_bucket(b, l_hop_sets, target_size)
        refined_chunks.extend(splits)

    # Step 2: Combine with remaining small buckets
    all_chunks = refined_chunks + small

    # Step 3: Greedy merge small chunks to get exactly m clusters
    final_clusters = greedy_merge_small_buckets(all_chunks, target_num_clusters, target_size)

    return final_clusters




########################################
# 4. Full Interface
########################################

def cluster_nodes_by_shared_neighbors(test_nodes, l, edge_index, m, method='exact', num_perm=128, threshold=0.5):
    """
    method: 'exact' or 'lsh'
    """
    l_hop_sets = get_all_l_hop_sets(test_nodes, l, edge_index)

    if method == 'exact':
        clusters = cluster_nodes_exact_safe(test_nodes, l_hop_sets, m)
        clusters = structure_preserving_balance(clusters, m)
    elif method == 'lsh':
        buckets = lsh_clustering_unique(test_nodes, l_hop_sets, threshold=threshold, num_perm=num_perm)
        clusters = refined_balance_lsh_clusters(buckets, m, l_hop_sets, tolerance=0.1)
    else:
        raise ValueError("Invalid method: choose 'exact' or 'lsh'")

    return clusters, l_hop_sets

########################################
# Visualization
########################################

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_clusters_with_links(test_nodes, clusters, l_hop_sets, shared_threshold=0.3, layout='spring'):
    """
    Visualize clustered nodes with edges based on shared l-hop neighbors.

    Parameters:
    - test_nodes: list of node indices
    - clusters: list of clusters (list of list of nodes)
    - l_hop_sets: dict mapping node to its l-hop neighbor set
    - shared_threshold: minimum Jaccard similarity to draw an edge
    - layout: 'spring' (default) or 'kamada_kawai'
    """

    # Build node2cluster map
    node2cluster = {}
    for cluster_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            node2cluster[node] = cluster_id

    # Build Graph
    G = nx.Graph()
    for node in test_nodes:
        G.add_node(node, cluster=node2cluster[node])

    # Add edges based on shared l-hop neighbors
    for i, u in enumerate(test_nodes):
        for j, v in enumerate(test_nodes):
            if i >= j:
                continue
            set_u = l_hop_sets[u]
            set_v = l_hop_sets[v]
            if not set_u or not set_v:
                continue
            jaccard = len(set_u & set_v) / len(set_u | set_v)
            if jaccard >= shared_threshold:
                G.add_edge(u, v, weight=jaccard)

    # Select layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Layout must be 'spring' or 'kamada_kawai'.")

    # Draw nodes
    plt.figure(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i % cmap.N) for i in range(len(clusters))]
    node_colors = [colors[node2cluster[n]] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, edgecolors='k')

    # Draw edges
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w * 3 for w in edge_weights])

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Legend
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=colors[i % cmap.N], label=f'Cluster {i}')
        for i in range(len(clusters))
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('Node Clusters with Shared l-Hop Neighbors')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



########################################
# Example Usage
########################################

config = load_config("config.yaml")
L = config['L']
data_name = config['data_name']
m = config['m']
random_seed = config['random_seed']
set_seed(random_seed)
data = dataset_func(config)

test_nodes = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))

# Cluster into 5 clusters
start_time = time.time()
clusters, l_hop_sets = cluster_nodes_by_shared_neighbors(
    test_nodes=test_nodes,
    l=L,
    edge_index=data.edge_index,
    m=m,
    method='exact',  # 'lsh' or 'exact'
    num_perm=256,
    threshold=0.7
)
end_time = time.time()
print(f'Partition Time: {end_time - start_time:.2f} seconds')

# for idx, cluster in enumerate(clusters):
#     print(f"Cluster {idx}: {cluster}")

torch.save(clusters, './datasets/{}/partition.pt'.format(data_name))

# Suppose you already have:
# - test_nodes (list of node indices)
# - clusters (list of clusters)
# - l_hop_sets (dict from earlier get_all_l_hop_sets)

# visualize_clusters_with_links(
#     test_nodes=test_nodes,
#     clusters=clusters,
#     l_hop_sets=l_hop_sets,
#     shared_threshold=0.3,   # adjust threshold: higher = stricter connections
#     layout='spring'         # or 'kamada_kawai'
# )

