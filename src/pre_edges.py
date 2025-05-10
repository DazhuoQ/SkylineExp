from src.utils import *

config = load_config("config.yaml")
data_name = config['data_name']
random_seed = config['random_seed']
L = config['L']
set_seed(random_seed)
data = dataset_func(config)

import torch
import torch_geometric
import os
import multiprocessing
from tqdm import tqdm
from collections import deque, defaultdict

# Set multiprocessing method early (important for MacOS/Linux)
multiprocessing.set_start_method('fork', force=True)


# Your original function (lightly optimized)
def get_edge_sets_by_hop(vt, G, L):
    edge_index = G.edge_index

    # Get l-hop subgraph
    node_idx, edge_index_sub, _, original_edge_mask = torch_geometric.utils.k_hop_subgraph(
        vt, L, edge_index, relabel_nodes=False
    )
    ori_mask = original_edge_mask
    selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
    subg_size = selected_edge_positions.size(0)

    # Build adjacency list for fast BFS
    adj_list = defaultdict(list)
    for edge_idx in selected_edge_positions:
        src, dst = edge_index[:, edge_idx]
        adj_list[src.item()].append(dst.item())
        adj_list[dst.item()].append(src.item())

    # BFS to compute hop distances
    hop_distances = {node.item(): float('inf') for node in node_idx}
    hop_distances[vt] = 0
    queue = deque([vt])

    while queue:
        current_node = queue.popleft()
        current_hop = hop_distances[current_node]
        for neighbor in adj_list[current_node]:
            if hop_distances[neighbor] == float('inf'):
                hop_distances[neighbor] = current_hop + 1
                queue.append(neighbor)

    # Group edges by hop
    edges_by_hop = defaultdict(list)
    edge_masks_by_hop = {}
    for edge_idx in selected_edge_positions:
        src, dst = edge_index[:, edge_idx]
        src_hop = hop_distances[src.item()]
        dst_hop = hop_distances[dst.item()]
        edge_hop = min(src_hop, dst_hop) + 1
        edges_by_hop[edge_hop].append(edge_idx.item())

    for hop in range(1, L + 2):
        mask = original_edge_mask.clone()
        if hop in edges_by_hop:
            for future_hop in range(hop + 1, L + 2):
                for edge_idx in edges_by_hop[future_hop]:
                    mask[edge_idx] = False
        edge_masks_by_hop[hop] = mask

    return edges_by_hop, edge_masks_by_hop, subg_size, ori_mask


# Wrapper for a single node (for multiprocessing)
def precompute_single_node(args):
    vt, G, L = args
    edges_by_hop, edge_masks_by_hop, subg_size, ori_mask = get_edge_sets_by_hop(vt, G, L)
    return {
        'edges_by_hop': edges_by_hop,
        'edge_masks_by_hop': {k: v.cpu() for k, v in edge_masks_by_hop.items()},
        'subg_size': subg_size,
        'ori_mask': ori_mask.cpu()
    }


# Batch precompute and save
def precompute_in_batches(G, list_of_nodes, L, num_workers=8, batch_size=100, save_dir='precomputed/'):
    os.makedirs(save_dir, exist_ok=True)
    
    batches = [list_of_nodes[i:i+batch_size] for i in range(0, len(list_of_nodes), batch_size)]
    
    for batch_idx, batch_nodes in enumerate(tqdm(batches, desc="Precomputing batches")):
        batch_results = {}
        args = [(vt, G, L) for vt in batch_nodes]

        with multiprocessing.Pool(num_workers) as pool:
            for vt, result in zip(batch_nodes, pool.map(precompute_single_node, args)):
                batch_results[vt] = result

        # Save this batch
        save_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
        torch.save(batch_results, save_path)


# Load all batches later
def load_precomputed(save_dir='precomputed/'):
    precomputed_data = {}
    for fname in sorted(os.listdir(save_dir)):
        if fname.endswith('.pt'):
            batch_data = torch.load(os.path.join(save_dir, fname))
            precomputed_data.update(batch_data)
    return precomputed_data




# Assume you have a PyG graph: data.edge_index

center_nodes = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))
max_1hop_neighbors = 10000  # Skip nodes with >1000 neighbors in 1-hop
num_workers = 8  # Number of parallel processes
batch_size = 100
save_dir = './precomputed/{}'.format(data_name)

# Example:
# G: your PyG graph object
# list_of_nodes: list of node ids to precompute
# L: number of hops

precompute_in_batches(data, center_nodes, L, num_workers=num_workers, batch_size=batch_size, save_dir=save_dir)

# Later
# precomputed_data = load_precomputed(save_dir)
# print(type(precomputed_data))
# print(precomputed_data[0].keys())
# print(precomputed_data[0]['subg_size'])
