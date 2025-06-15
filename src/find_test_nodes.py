# 自适应导入utils模块
try:
    from utils import *  # 当在src目录内运行时
except ImportError:
    try:
        from src.utils import *  # 当在项目根目录运行时
    except ImportError:
        raise ImportError("无法导入utils模块，请确保在正确的目录中运行脚本")
import os
import torch
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
import random
from tqdm import tqdm
from collections import deque

config_path = "config.yaml"
if not os.path.exists(config_path):
    config_path = "../config.yaml"
    
if not os.path.exists(config_path):
    raise FileNotFoundError("找不到config.yaml文件，请确保它在当前目录或上一级目录中")

config = load_config(config_path)
data_name = config['data_name']
random_seed = config['random_seed']
L = config['L']
m = config.get('m', 8)  # 获取配置中的m值，默认为8
group_size = 30  # 每组节点数量
set_seed(random_seed)
data = dataset_func(config)

# 为BAHouse创建目录
os.makedirs(f'./datasets/{data_name}', exist_ok=True)

# 针对BAHouse数据集的特殊处理
if data_name == 'BAHouse':
    print(f"检测到大型数据集BAHouse，使用随机采样方法...")
    num_nodes = data.num_nodes
    print(f"BAHouse数据集共有 {num_nodes} 个节点")
    
    # 保持与原代码相同的采样逻辑
    target_samples = m * group_size  # 与原来逻辑一致，保持m组，每组group_size个节点
    print(f"目标采样数量: {target_samples} 个节点 ({m} 组，每组 {group_size} 个节点)")
    
    # 随机采样节点
    nodes_selected = random.sample(range(num_nodes), min(target_samples, num_nodes))
    nodes_selected = sorted(nodes_selected)  # 排序以便更好地理解
    
    print(f"已随机选择 {len(nodes_selected)} 个测试节点")
    
    # 保存测试节点
    torch.save(nodes_selected, f'./datasets/{data_name}/test_nodes.pt')
    print(f"测试节点已保存到 ./datasets/{data_name}/test_nodes.pt")
    
    # 如果需要生成partition.pt文件
    if config.get('method', '') == 'share_cluster_para':
        print("创建简单分区文件...")
        # 随机打乱节点顺序并分组
        shuffled = nodes_selected.copy()
        random.shuffle(shuffled)
        partitions = []
        nodes_per_partition = len(shuffled) // m
        for i in range(m):
            start_idx = i * nodes_per_partition
            end_idx = start_idx + nodes_per_partition if i < m - 1 else len(shuffled)
            partitions.append([int(n) for n in shuffled[start_idx:end_idx]])
        
        torch.save(partitions, f'./datasets/{data_name}/partition.pt')
        print(f"分区文件已保存到 ./datasets/{data_name}/partition.pt")
    
    # 提前退出，不执行原来的代码
    import sys
    sys.exit(0)

# 针对arxiv数据集的高效处理
elif data_name == 'arxiv':
    print(f"检测到大型数据集arxiv，使用高效聚类方法...")
    num_nodes = data.num_nodes
    print(f"arxiv数据集共有 {num_nodes} 个节点")
    
    # 将图转换为NetworkX图以便处理
    G_nx = to_networkx(data, to_undirected=True)
    
    # 1. 首先选择高度节点作为种子
    # 计算度中心性（仅取样部分节点以提高效率）
    sample_size = min(10000, num_nodes)  # 最多取样10000个节点计算度
    sampled_nodes = random.sample(range(num_nodes), sample_size)
    node_degrees = {node: G_nx.degree(node) for node in sampled_nodes}
    
    # 按度排序
    sorted_nodes = sorted(node_degrees.keys(), key=lambda x: -node_degrees[x])
    
    # 2. 选择分散的种子节点
    seeds = []
    min_distance = 3  # 种子之间的最小距离
    for candidate in sorted_nodes:
        # 检查与现有种子的距离
        too_close = False
        for seed in seeds:
            # 使用简单的最短路径计算
            try:
                path_length = nx.shortest_path_length(G_nx, source=seed, target=candidate)
                if path_length < min_distance:
                    too_close = True
                    break
            except nx.NetworkXNoPath:
                # 如果节点之间没有路径，则认为它们足够远
                pass
        
        if not too_close:
            seeds.append(candidate)
            if len(seeds) >= m:
                break
    
    print(f"选择了 {len(seeds)} 个种子节点")
    
    # 3. 围绕每个种子生成组
    nodes_selected = []
    partitions = []
    
    for seed in seeds:
        # 使用BFS从种子节点向外扩展
        group = [seed]
        visited = {seed}
        queue = deque([seed])
        
        while len(group) < group_size and queue:
            current = queue.popleft()
            for neighbor in G_nx.neighbors(current):
                if neighbor not in visited and len(group) < group_size:
                    visited.add(neighbor)
                    group.append(neighbor)
                    queue.append(neighbor)
        
        # 如果通过BFS无法获得足够的节点，随机添加剩余节点
        if len(group) < group_size:
            remaining = group_size - len(group)
            available = [n for n in range(num_nodes) if n not in visited]
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                group.extend(additional)
        
        group = [int(n) for n in group]  # 确保节点是整数
        nodes_selected.extend(group)
        partitions.append(group)
    
    print(f"共选择了 {len(nodes_selected)} 个节点")
    
    # 保存测试节点
    torch.save(nodes_selected, f'./datasets/{data_name}/test_nodes.pt')
    print(f"测试节点已保存到 ./datasets/{data_name}/test_nodes.pt")
    
    # 如果需要生成partition.pt文件
    if config.get('method', '') == 'share_cluster_para':
        torch.save(partitions, f'./datasets/{data_name}/partition.pt')
        print(f"分区文件已保存到 ./datasets/{data_name}/partition.pt")
    
    # 提前退出，不执行原来的代码
    import sys
    sys.exit(0)

# 原有代码继续执行，用于其他数据集
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
