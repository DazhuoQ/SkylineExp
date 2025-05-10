from src.utils import *

config = load_config("config.yaml")
data_name = config['data_name']
random_seed = config['random_seed']
L = config['L']
set_seed(random_seed)
data = dataset_func(config)

import torch
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
import random
from tqdm import tqdm
from collections import deque

def bfs_distances(G, nodes, max_hops):
    distances = {node: {} for node in nodes}
    for node in tqdm(nodes, desc="Precomputing BFS distances"):
        visited = {node: 0}
        queue = deque([node])
        while queue:
            u = queue.popleft()
            if visited[u] >= max_hops:
                continue
            for v in G.neighbors(u):
                if v not in visited:
                    visited[v] = visited[u] + 1
                    queue.append(v)
        distances[node] = visited
    return distances

def select_seeds_high_degree_lhop_limit(G, bfs_dists, l_hop_sizes, m, min_hop_distance, max_lhop_size):
    # Rank nodes by degree (high degree first)
    node_degrees = dict(G.degree())
    sorted_nodes = sorted(node_degrees, key=lambda x: -node_degrees[x])

    seeds = []
    for candidate in sorted_nodes:
        if l_hop_sizes[candidate] > max_lhop_size:
            continue  # Skip too big nodes
        if all(bfs_dists[seed].get(candidate, 1e9) >= min_hop_distance for seed in seeds):
            seeds.append(candidate)
            if len(seeds) >= m:
                break
    return seeds


def grow_group_exact_size_with_lhop_limit(G, seed, assigned, target_size, lhop_sizes, max_lhop_size):
    group = [seed]
    assigned.add(seed)
    queue = deque([seed])
    
    while len(group) < target_size:
        if not queue:
            # If the queue is empty but group not full, select random unassigned node satisfying lhop constraint
            unassigned = [n for n in G.nodes if n not in assigned and lhop_sizes.get(n, 1e9) <= max_lhop_size]
            if not unassigned:
                break  # No more eligible nodes
            next_node = random.choice(unassigned)
            queue.append(next_node)
        
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in assigned and lhop_sizes.get(v, 1e9) <= max_lhop_size:
                group.append(v)
                assigned.add(v)
                queue.append(v)
                if len(group) >= target_size:
                    break
    return group


def compute_lhop_sizes(data, all_nodes, l):
    edge_index = data.edge_index
    lhop_sizes = {}
    for node in tqdm(all_nodes, desc="Computing l-hop subgraph sizes"):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=node,
            num_hops=l,
            edge_index=edge_index,
            relabel_nodes=False
        )
        lhop_sizes[node] = len(subset)  # number of nodes including center
    return lhop_sizes

def cluster_by_bfs_exact(data, m=5, group_size=100, min_hop_distance=3, max_hops=5, l=2, max_lhop_size=200):
    G_nx = to_networkx(data, to_undirected=True)
    all_nodes = list(range(data.num_nodes))

    # Precompute
    bfs_dists = bfs_distances(G_nx, all_nodes, max_hops=max_hops)
    lhop_sizes = compute_lhop_sizes(data, all_nodes, l=l)

    seeds = select_seeds_high_degree_lhop_limit(
        G_nx, bfs_dists, lhop_sizes,
        m=m, min_hop_distance=min_hop_distance, max_lhop_size=max_lhop_size
    )
    if len(seeds) < m:
        print(f"Warning: Only {len(seeds)} seeds could be selected with min_hop_distance={min_hop_distance}")
    print(f"Selected seeds: {seeds}")

    assigned = set()
    groups = {}

    # Step 2: Grow groups around seeds
    assigned = set()
    groups = {}
    for seed in seeds:
        groups[seed] = grow_group_exact_size_with_lhop_limit(
            G_nx, seed, assigned,
            target_size=group_size,
            lhop_sizes=lhop_sizes,
            max_lhop_size=max_lhop_size
        )

    return groups


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_groups_with_seeds(data, groups, layout='spring', node_size=100, seed_node_size=300):
    G_nx = to_networkx(data, to_undirected=True)
    
    # Only show selected nodes
    selected_nodes = []
    for group in groups.values():
        selected_nodes.extend(group)
    selected_nodes = set(selected_nodes)

    subG = G_nx.subgraph(selected_nodes)

    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(subG, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(subG)
    elif layout == 'spectral':
        pos = nx.spectral_layout(subG)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Assign colors
    group_color_map = {}
    colors = plt.cm.get_cmap('tab10', len(groups))

    # Separate seeds and normal nodes
    seed_nodes = []
    normal_nodes = []
    node_colors = []
    for idx, (seed, nodes) in enumerate(groups.items()):
        for node in nodes:
            group_color_map[node] = colors(idx)
            if node == seed:
                seed_nodes.append(node)
            else:
                normal_nodes.append(node)

    # Prepare node colors
    normal_colors = [group_color_map[node] for node in normal_nodes]
    seed_colors = [group_color_map[node] for node in seed_nodes]

    # Draw
    plt.figure(figsize=(9, 9))
    nx.draw_networkx_edges(subG, pos, alpha=0.2)
    nx.draw_networkx_nodes(subG, pos, nodelist=normal_nodes, node_color=normal_colors, node_size=node_size)
    nx.draw_networkx_nodes(
        subG, pos,
        nodelist=seed_nodes,
        node_color=seed_colors,
        node_size=seed_node_size,
        edgecolors='black',  # black outline
        linewidths=1.5
    )
    plt.axis('off')
    plt.title(f"Visualization of {len(groups)} groups (highlight seeds)", fontsize=14)
    plt.show()


m = 8
group_size = 30


groups = cluster_by_bfs_exact(data, m=m, group_size=group_size, min_hop_distance=10, l=L, max_lhop_size=200)

nodes_selected = []
for i, (seed, nodes) in enumerate(groups.items()):
    clean_nodes = [int(i) for i in nodes]
    nodes_selected.extend(clean_nodes)
    # print(f"Group {i}: seed {seed}, {len(nodes)} nodes")
# print(nodes_selected)

# visualize_groups_with_seeds(data, groups, layout='spring', node_size=50, seed_node_size=300)

torch.save(nodes_selected, './datasets/{}/test_nodes.pt'.format(data_name))
