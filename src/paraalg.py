from src.utils import *
from src.model import get_model
from src.apxalgop_para import ParaApxSXOP
from src.apxalgi_para import ParaApxSXI
from src.divalg_para import ParaDivSX
from src.apxalgop_share import ShareApxSXOP

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import time
from tqdm import tqdm


def run_algorithm_subset(args):
    G, model, VT_subset, k, L, epsilon, beta, alpha, counter, lock = args
    # algorithm = ParaApxSXOP(G=G, model=model, VT=VT_subset, k=k, L=L, epsilon=epsilon)
    # algorithm = ParaApxSXI(G=G, model=model, VT=VT_subset, k=k, L=L, epsilon=epsilon)
    # algorithm = ParaDivSX(G=G, model=model, VT=VT_subset, k=k, L=L, epsilon=epsilon, beta=beta, alpha=alpha)
    algorithm = ShareApxSXOP(G=G, model=model, VT=VT_subset, k=k, L=L, epsilon=epsilon)

    results = []
    for node in VT_subset:
        algorithm.generate_k_skylines(node)  # assume your algorithm has a method processing a single node
        results.extend(algorithm.k_sky_lst)
        with lock:
            counter.value += 1  # safely increment processed node count

    algorithm.IPF()
    return results, algorithm.ipf


def monitor_progress(counters, total_nodes, interval=0.5):
    pbar_list = [tqdm(total=total, desc=f'Thread-{idx}', position=idx) for idx, total in enumerate(total_nodes)]
    
    completed = [0] * len(counters)
    while sum(completed) < sum(total_nodes):
        for idx, counter in enumerate(counters):
            pbar_list[idx].n = counter.value
            pbar_list[idx].refresh()
            completed[idx] = counter.value
        time.sleep(interval)

    for pbar in pbar_list:
        pbar.close()


def preprocessing(G, model, VT, k, L, epsilon, beta, alpha, m):
    VT_subsets = np.array_split(VT, m)
    # print(f'VT_subsets:{VT_subsets}')
    nxG = to_networkx(G, to_undirected=True)
    VT_subsets = compute_new_VT_subset(nxG, VT_subsets)
    # print(f'VT_subsets:{VT_subsets}')

    manager = multiprocessing.Manager()
    counters = [manager.Value('i', 0) for _ in range(m)]
    lock = manager.Lock()

    args = [
        (G, model, subset, k, L, epsilon, beta, alpha, counters[idx], lock)
        for idx, subset in enumerate(VT_subsets)
    ]

    total_nodes = [len(subset) for subset in VT_subsets]
    return args, counters, total_nodes


# Function to rank nodes based on core number (descending order)
def rank_nodes_by_core(subgraph):
    core_numbers = nx.core_number(subgraph)
    ranked_nodes = sorted(core_numbers.items(), key=lambda x: x[1], reverse=True)
    ranked_node_list = [node for node, core in ranked_nodes]
    return ranked_node_list


def compute_new_VT_subset(nxG, VT_subsets):
    # Rank nodes in each subset
    ranked_VT_subsets = []
    for subset in VT_subsets:
        nodes_list = subset.tolist()
        subgraph = nxG.subgraph(nodes_list)
        ranked_nodes = rank_nodes_by_core(subgraph)
        ranked_VT_subsets.append(torch.tensor(ranked_nodes))
    return ranked_VT_subsets


def parallelize_algorithm(args, counters, total_nodes, m):

    # Start monitoring process
    monitor_process = multiprocessing.Process(target=monitor_progress, args=(counters, total_nodes))
    monitor_process.start()

    start_time = time.time()
    results = []
    ipf_results = []
    with ProcessPoolExecutor(max_workers=m) as executor:
        futures = [executor.submit(run_algorithm_subset, arg) for arg in args]
        for future in futures:
            result, ipf = future.result()
            results.extend(result)
            ipf_results.append(ipf)
    end_time = time.time()
    
    monitor_process.join()

    print(f"Parallel execution time: {end_time - start_time:.2f} seconds")
    return results, ipf_results



def main():

    # print(multiprocessing.cpu_count())

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

    set_seed(random_seed)

    # Get input graph
    data = dataset_func(config)

    # Get the VT
    VT = torch.tensor(VT)
    # VT = VT + 1000

    # Ready the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config)
    model.load_state_dict(torch.load('models/{}_{}_model.pth'.format(data_name, model_name), weights_only=False, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    exp_name = 'para'
    if exp_name == 'para':

        args, counters, total_nodes = preprocessing(G=data, model=model, VT=VT, k=k, L=L, epsilon=epsilon, beta=beta, alpha=alpha, m=m)
        results, ipf_results = parallelize_algorithm(args, counters, total_nodes, m=m)

        # print(results)
        print(np.mean(ipf_results))

if __name__ == "__main__":
    main()