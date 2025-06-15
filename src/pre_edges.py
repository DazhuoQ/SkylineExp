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


def get_edge_sets_by_hop(vt, G, L, max_subgraph_size=50000):
    edge_index = G.edge_index
    
    # Get node degree for early stopping if the node is too large
    node_degree = torch_geometric.utils.degree(edge_index[0], num_nodes=G.num_nodes)
    if node_degree[vt] > max_subgraph_size:
        return None, None, 0, None  # Skip nodes with too many connections
    
    # 使用纯PyG实现，移除复杂的cuGraph逻辑以提高稳定性
    try:
        # 使用PyG的k_hop_subgraph方法获取子图
        node_idx, edge_index_sub, _, original_edge_mask = torch_geometric.utils.k_hop_subgraph(
            vt, L, edge_index, relabel_nodes=False, num_nodes=G.num_nodes
        )
        
        # 检查子图大小
        if original_edge_mask.sum() > max_subgraph_size:
            print(f"Subgraph for node {vt} too large: {original_edge_mask.sum()} edges. Skipping...")
            return None, None, 0, None
        
        ori_mask = original_edge_mask
        selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
        
        # 处理只有一条边的情况
        if selected_edge_positions.dim() == 0:
            selected_edge_positions = selected_edge_positions.unsqueeze(0)
        
        subg_size = selected_edge_positions.size(0)
        
        # 构建邻接表用于快速BFS
        adj_list = defaultdict(set)
        for edge_idx in selected_edge_positions:
            src, dst = edge_index[:, edge_idx]
            src_item, dst_item = src.item(), dst.item()
            adj_list[src_item].add(dst_item)
            adj_list[dst_item].add(src_item)
        
        # 使用BFS计算hop距离
        hop_distances = {}
        hop_distances[vt] = 0
        queue = deque([vt])
        visited = {vt}
        
        while queue:
            current_node = queue.popleft()
            current_hop = hop_distances[current_node]
            
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    hop_distances[neighbor] = current_hop + 1
                    if current_hop + 1 < L:  # 只在需要时继续BFS
                        queue.append(neighbor)
        
        # 按hop对边进行分组
        edges_by_hop = defaultdict(list)
        for edge_idx in selected_edge_positions:
            src, dst = edge_index[:, edge_idx]
            src_item, dst_item = src.item(), dst.item()
            src_hop = hop_distances.get(src_item, float('inf'))
            dst_hop = hop_distances.get(dst_item, float('inf'))
            edge_hop = min(src_hop, dst_hop) + 1
            if edge_hop <= L + 1:
                edges_by_hop[edge_hop].append(edge_idx.item())
        
        # 创建边掩码
        edge_masks_by_hop = {}
        cumulative_edges = []
        for hop in range(1, L + 2):
            if hop in edges_by_hop:
                cumulative_edges.extend(edges_by_hop[hop])
                mask = torch.zeros_like(original_edge_mask)
                mask[cumulative_edges] = True
                edge_masks_by_hop[hop] = mask
        
        return edges_by_hop, edge_masks_by_hop, subg_size, ori_mask
    
    except Exception as e:
        print(f"Error processing node {vt}: {str(e)}")
        return None, None, 0, None


# 改进的单节点预计算包装函数
def precompute_single_node(args):
    vt, G, L, max_subgraph_size = args
    try:
        result = get_edge_sets_by_hop(vt, G, L, max_subgraph_size)
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask = result
        
        if edges_by_hop is None:  # 跳过太大或处理失败的节点
            print(f"Skipping node {vt} - processing failed or too large")
            return vt, None
            
        return vt, {
            'edges_by_hop': edges_by_hop,
            'edge_masks_by_hop': {k: v.cpu() for k, v in edge_masks_by_hop.items()},
            'subg_size': subg_size,
            'ori_mask': ori_mask.cpu() if ori_mask is not None else None
        }
    except Exception as e:
        print(f"Unexpected error processing node {vt}: {str(e)}")
        return vt, None


# 改进的批量预计算和保存函数
def precompute_in_batches(G, list_of_nodes, L, num_workers=4, batch_size=50, 
                         save_dir='precomputed/', max_subgraph_size=5000):
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录失败的节点
    failed_nodes_file = os.path.join(save_dir, "failed_nodes.txt")
    failed_nodes = []
    
    # 过滤掉度数太大的节点
    if hasattr(G, 'num_nodes'):
        degree = torch_geometric.utils.degree(G.edge_index[0], num_nodes=G.num_nodes)
        filtered_nodes = [node for node in list_of_nodes if degree[node] <= max_subgraph_size]
        skipped_nodes = set(list_of_nodes) - set(filtered_nodes)
        failed_nodes.extend(skipped_nodes)
        print(f"Filtered out {len(list_of_nodes) - len(filtered_nodes)} nodes with degree > {max_subgraph_size}")
        list_of_nodes = filtered_nodes
    
    batches = [list_of_nodes[i:i+batch_size] for i in range(0, len(list_of_nodes), batch_size)]
    
    for batch_idx, batch_nodes in enumerate(tqdm(batches, desc="Precomputing batches")):
        batch_results = {}
        args = [(vt, G, L, max_subgraph_size) for vt in batch_nodes]
        
        # 使用进程池处理批次
        with multiprocessing.Pool(num_workers) as pool:
            for vt, result in pool.map(precompute_single_node, args):
                if result is not None:  # 跳过失败的节点
                    batch_results[vt] = result
                else:
                    failed_nodes.append(vt)
        
        # 保存这个批次（如果不为空）
        if batch_results:
            save_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
            try:
                torch.save(batch_results, save_path)
                print(f"Saved batch {batch_idx} with {len(batch_results)} nodes")
            except Exception as e:
                print(f"Error saving batch {batch_idx}: {str(e)}")
                # 记录整个批次的失败节点
                failed_nodes.extend(batch_nodes)
        
        # 清理内存
        del batch_results
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 保存失败的节点列表
    with open(failed_nodes_file, 'w') as f:
        for node in failed_nodes:
            f.write(f"{node}\n")
    print(f"Saved list of {len(failed_nodes)} failed nodes to {failed_nodes_file}")


# 增强的加载函数，支持容错
def load_precomputed(save_dir='precomputed/', batch_size=10):
    """加载预计算数据，带错误处理和日志"""
    all_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pt')])
    precomputed_data = {}
    failed_files = []
    
    print(f"Loading {len(all_files)} batch files from {save_dir}")
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        
        for fname in batch_files:
            try:
                file_path = os.path.join(save_dir, fname)
                file_data = torch.load(file_path)
                precomputed_data.update(file_data)
            except Exception as e:
                print(f"Error loading file {fname}: {str(e)}")
                failed_files.append(fname)
        
        # 周期性报告进度
        if (i//batch_size) % 10 == 0:
            print(f"Loaded {i//batch_size} batches, current data size: {len(precomputed_data)} nodes")
    
    if failed_files:
        print(f"Warning: Failed to load {len(failed_files)} files: {failed_files[:5]}...")
    
    print(f"Successfully loaded data for {len(precomputed_data)} nodes")
    return precomputed_data


# 修改主执行代码，移除cuGraph相关逻辑简化实现
if __name__ == "__main__":
    print("Using PyG implementation for graph processing")
    
    # 加载测试节点
    center_nodes = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))
    max_subgraph_size = 200  # 跳过邻居太多的节点
    num_workers = 8  # 减少并行进程数量提高稳定性
    batch_size = 50  # 较小的批次大小以减少内存使用
    save_dir = './precomputed/{}'.format(data_name)

    precompute_in_batches(data, center_nodes, L, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        save_dir=save_dir,
                        max_subgraph_size=max_subgraph_size)

    # 验证加载过程
    print("Verifying data loading...")
    sample_data = load_precomputed(save_dir, batch_size=5)
    print(f"Verification complete: loaded {len(sample_data)} nodes")
