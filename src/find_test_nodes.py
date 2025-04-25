import torch
from torch_geometric.utils import k_hop_subgraph
from datasketch import MinHash, MinHashLSH
import random
from typing import List, Set, Dict

from src.utils import *

# -----------------------------
# Step 1: Extract l-hop neighborhood
# -----------------------------
def get_l_hop_neighbors(node_idx: int, l: int, edge_index: torch.Tensor) -> Set[int]:
    subset, _, _, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=l,
        edge_index=edge_index,
        relabel_nodes=False
    )
    return set(subset.tolist()) - {node_idx}  # Exclude center node

# -----------------------------
# Step 2: Create MinHash from a neighborhood
# -----------------------------
def get_minhash(neighbors: Set[int], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for n in neighbors:
        m.update(str(n).encode('utf8'))
    return m

# -----------------------------
# Step 3: Perform LSH to group similar nodes
# -----------------------------
def lsh_clustering(
    test_nodes: List[int],
    l: int,
    edge_index: torch.Tensor,
    threshold: float = 0.5,
    num_perm: int = 128
) -> List[List[int]]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    node_hashes: Dict[int, MinHash] = {}

    print("Building MinHash and inserting into LSH index...")
    for node in test_nodes:
        neighbors = get_l_hop_neighbors(node, l, edge_index)
        mh = get_minhash(neighbors, num_perm)
        lsh.insert(str(node), mh)
        node_hashes[node] = mh

    print("Grouping nodes into buckets...")
    buckets: Dict[str, List[int]] = {}
    for node, mh in node_hashes.items():
        key = tuple(sorted(lsh.query(mh)))
        key_str = str(key)
        if key_str not in buckets:
            buckets[key_str] = []
        buckets[key_str].append(node)

    print(f"Number of clusters found: {len(buckets)}")
    return list(buckets.values())

# -----------------------------
# Step 4: Select n nodes and split into m balanced clusters
# -----------------------------
def balanced_merge_lsh_clusters(
    clusters: List[List[int]],
    n: int,
    m: int
) -> List[List[int]]:
    all_nodes = [node for cluster in clusters for node in cluster]
    if n > len(all_nodes):
        print(f"Warning: requested n={n} exceeds available {len(all_nodes)} nodes. Using all.")
        n = len(all_nodes)

    selected_nodes = random.sample(all_nodes, n)
    random.shuffle(selected_nodes)

    size = n // m
    subsets = [selected_nodes[i * size:(i + 1) * size] for i in range(m)]

    # Distribute leftovers
    leftovers = selected_nodes[m * size:]
    for i, node in enumerate(leftovers):
        subsets[i % m].append(node)

    print(f"Created {m} clusters with ~{n//m} nodes each (plus remainders).")
    return subsets

# -----------------------------
# Final Wrapper
# -----------------------------
def cluster_test_nodes_large(
    test_nodes: List[int],
    edge_index: torch.Tensor,
    l: int,
    n: int,
    m: int,
    threshold: float = 0.5,
    num_perm: int = 128
) -> List[List[int]]:
    print(f"Starting clustering on {len(test_nodes)} test nodes...")
    lsh_clusters = lsh_clustering(test_nodes, l, edge_index, threshold=threshold, num_perm=num_perm)
    return balanced_merge_lsh_clusters(lsh_clusters, n, m)



config = load_config("config.yaml")
L = config['L']
data_name = config['data_name']
data_size = config['data_size']
m = config['m']
random_seed = config['random_seed']
set_seed(random_seed)
data = dataset_func(config)


test_nodes = list(range(data_size))

# Assume test_nodes is a list of node indices, and edge_index is from your PyG graph
VT_subsets = cluster_test_nodes_large(
    test_nodes=test_nodes,
    edge_index=data.edge_index,
    l=L,               # l-hop
    n=1000,           # total nodes to sample
    m=m,              # number of clusters
    threshold=0.5,     # LSH Jaccard threshold
    num_perm=128       # MinHash permutations
)
import itertools
VT_subsets = list(itertools.chain(*VT_subsets))
# VT_subsets = [torch.tensor(i) for i in VT_subsets]
# print(VT_subsets)
torch.save(VT_subsets, './datasets/{}/test_nodes.pt'.format(data_name))
