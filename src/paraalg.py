from src.utils import *
from src.model import get_model
from src.apxalgop_para import ParaApxSXOP
from src.apxalgop_share import ShareApxSXOP

import torch.multiprocessing as mp
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import time
import matplotlib.pyplot as plt
import random

mp.set_start_method("spawn", force=True)

def visualize_node_groups(data, node_groups, node_color_map=None, layout='spring'):
    G = to_networkx(data, to_undirected=True)
    selected_nodes = [int(node) for group in node_groups for node in group]
    G_sub = G.subgraph(selected_nodes)
    if node_color_map is None:
        cmap = plt.colormaps.get_cmap('tab10')
        node_color_map = [cmap(i) for i in range(len(node_groups))]
    node_colors = []
    node_to_color = {}
    for idx, group in enumerate(node_groups):
        for node in group:
            node_to_color[int(node)] = node_color_map[idx]
    for node in G_sub.nodes():
        node_colors.append(node_to_color.get(int(node), 'gray'))
    if layout == 'spring':
        pos = nx.spring_layout(G_sub, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_sub)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G_sub)
    else:
        pos = nx.spring_layout(G_sub)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=200)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    plt.axis('off')
    plt.show()

def worker(rank, VT_subset, results_list, ipf_list, lookup_list, overlap_list,
           method, G, model, data_name, k, L, epsilon, precomputed_data):
    import torch
    torch.set_num_threads(1)
    VT_subset = [int(node) for node in VT_subset]
    if method == 'simple_para':
        algorithm = ParaApxSXOP(G, model, data_name, VT_subset, k, L, epsilon, precomputed_data)
    elif method.startswith('share'):
        algorithm = ShareApxSXOP(G, model, data_name, VT_subset, k, L, epsilon, precomputed_data)

    local_results = []
    for node in VT_subset:
        algorithm.generate_k_skylines(int(node))
        local_results.extend(algorithm.k_sky_lst)

    algorithm.IPF()
    results_list[rank] = local_results
    ipf_list[rank] = algorithm.ipf
    lookup_list[rank] = len(algorithm.ranked_indices_set)
    overlap_list[rank] = algorithm.overlap_cnt

def rank_nodes_by_core(subgraph):
    core_numbers = nx.core_number(subgraph)
    ranked_nodes = sorted(core_numbers.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in ranked_nodes]

def compute_new_VT_subset(nxG, VT_subsets):
    ranked_VT_subsets = []
    for subset in VT_subsets:
        subgraph = nxG.subgraph(subset)
        ranked_nodes = rank_nodes_by_core(subgraph)
        ranked_VT_subsets.append(ranked_nodes)
    return ranked_VT_subsets

def preprocessing():
    global G, data_name, method, L, m
    if method == 'share_cluster_para':
        VT_subsets = torch.load(f'./datasets/{data_name}/partition.pt').copy()
        nxG = to_networkx(G, to_undirected=True)
        VT_subsets = compute_new_VT_subset(nxG, VT_subsets)
        # visualize_node_groups(G, VT_subsets)
    else:
        VT = torch.load(f'./datasets/{data_name}/test_nodes.pt').copy()
        random.shuffle(VT)
        VT_subsets = np.array_split(VT, m)
        VT_subsets = [list(sub) for sub in VT_subsets]
        # visualize_node_groups(G, VT_subsets)
    return VT_subsets

def parallelize_algorithm_shared(VT_subsets, method, G, model, data_name, k, L, epsilon, precomputed_data):
    start_time = time.time()
    
    manager = mp.Manager()
    m = len(VT_subsets)
    results_list = manager.list([[] for _ in range(m)])
    ipf_list = manager.list([0.0] * m)
    lookup_list = manager.list([0] * m)
    overlap_list = manager.list([0] * m)

    processes = []
    for rank in range(m):
        p = mp.Process(target=worker, args=(
            rank, VT_subsets[rank],
            results_list, ipf_list, lookup_list, overlap_list,
            method, G, model, data_name, k, L, epsilon, precomputed_data
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    
    end_time = time.time()
    print(f'Total Computation Time (across processes): {end_time - start_time:.2f} seconds')
    print(f'Number of Processes: {m}')
    print(f'Algorithm Method: {method}')
    results = [r for sublist in results_list for r in sublist]
    return results, list(ipf_list), list(lookup_list), list(overlap_list)

def main():
    VT_subsets = preprocessing()
    results, ipf_results, lookup_results, overlap_results = parallelize_algorithm_shared(
        VT_subsets, method, G, model, data_name, k, L, epsilon, precomputed_data
    )
    print(f'IPF: {np.mean(ipf_results):.4f}')
    # print(f'lookup_results: {lookup_results}, total: {np.sum(lookup_results)}')
    # print(f'overlap_results: {overlap_results}, total: {np.sum(overlap_results)}')

if __name__ == "__main__":
    config = load_config("config.yaml")
    data_name = config['data_name']
    model_name = config['model_name']
    random_seed = config['random_seed']
    L = config['L']
    k = config['k']
    epsilon = config['epsilon']
    exp_name = config['exp_name']
    VT = config['VT']
    m = config['m']
    beta = config['beta']
    alpha = config['alpha']
    method = config['method']

    set_seed(random_seed)
    data = dataset_func(config)
    G = data

    if hasattr(G, 'edge_index'):
        G.edge_index = G.edge_index.share_memory_()
    if hasattr(G, 'x'):
        G.x = G.x.share_memory_()

    device = 'cpu'
    model = get_model(config)
    model.load_state_dict(torch.load(f'models/{data_name}_{model_name}_model.pth', map_location='cpu'))
    model.eval()
    model.share_memory()

    save_dir = f'./precomputed/{data_name}'
    precomputed_data = load_precomputed(save_dir)
    for k_pre, v in precomputed_data.items():
        if isinstance(v, torch.Tensor):
            precomputed_data[k_pre] = v.share_memory_()

    main()
