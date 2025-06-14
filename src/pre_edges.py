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

# 添加cuGraph相关导入
import numpy as np
try:
    import cudf
    import cupy as cp
    import cugraph
    CUGRAPH_AVAILABLE = True
except ImportError:
    CUGRAPH_AVAILABLE = False
    print("cuGraph not available. Falling back to CPU implementation.")

# Set multiprocessing method early (important for MacOS/Linux)
multiprocessing.set_start_method('fork', force=True)


# 添加PyG图转换为cuGraph图的函数
def pyg_to_cugraph(edge_index, num_nodes=None):
    """Convert PyG graph to cuGraph graph for faster processing"""
    if not CUGRAPH_AVAILABLE:
        return None
    
    try:
        # 创建cuDF DataFrame表示边
        src = cudf.Series(edge_index[0].cpu().numpy())
        dst = cudf.Series(edge_index[1].cpu().numpy())
        
        # 创建带权重的边DataFrame (使用1作为默认权重)
        df = cudf.DataFrame()
        df['src'] = src
        df['dst'] = dst
        df['weight'] = 1.0
        
        # 创建cuGraph图
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
        return G
    except Exception as e:
        print(f"Error converting to cuGraph: {e}")
        return None


# Your original function (optimized for large graphs)
def get_edge_sets_by_hop(vt, G, L, max_subgraph_size=50000):
    edge_index = G.edge_index
    
    # Get node degree for early stopping if the node is too large
    node_degree = torch_geometric.utils.degree(edge_index[0], num_nodes=G.num_nodes)
    if node_degree[vt] > max_subgraph_size:
        return None, None, 0, None  # Skip nodes with too many connections
    
    # 使用cuGraph进行加速计算 (如果可用)
    if CUGRAPH_AVAILABLE:
        try:
            # 转换为cuGraph图
            cug = pyg_to_cugraph(edge_index, num_nodes=G.num_nodes)
            if cug is not None:
                # 使用cuGraph的BFS获取k-hop信息
                df = cugraph.traversal.bfs(cug, vt, depth_limit=L)
                
                # 从结果中获取节点和距离
                nodes = df['vertex'].to_pandas().values
                distances = df['distance'].to_pandas().values
                
                # 创建节点到距离的映射
                hop_distances = {int(nodes[i]): int(distances[i]) for i in range(len(nodes))}
                
                # 为原始实现创建edge_mask
                original_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
                
                # 找出所有连接到我们k-hop子图的边
                for i in range(edge_index.size(1)):
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    if src in hop_distances and dst in hop_distances:
                        original_edge_mask[i] = True
                
                # 如果子图太大，跳过
                if original_edge_mask.sum() > max_subgraph_size:
                    print(f"Subgraph for node {vt} too large: {original_edge_mask.sum()} edges. Skipping...")
                    return None, None, 0, None
                
                ori_mask = original_edge_mask
                selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
                
                # 处理只有一条边的情况
                if selected_edge_positions.dim() == 0:
                    selected_edge_positions = selected_edge_positions.unsqueeze(0)
                
                subg_size = selected_edge_positions.size(0)
                
                # 按照hop对边进行分组
                edges_by_hop = defaultdict(list)
                for edge_idx in selected_edge_positions:
                    src, dst = edge_index[:, edge_idx]
                    src_hop = hop_distances.get(src.item(), float('inf'))
                    dst_hop = hop_distances.get(dst.item(), float('inf'))
                    edge_hop = min(src_hop, dst_hop) + 1
                    if edge_hop <= L + 1:
                        edges_by_hop[edge_hop].append(edge_idx.item())
                
                # 创建按hop划分的边掩码
                edge_masks_by_hop = {}
                for hop in range(1, L + 2):
                    if hop in edges_by_hop:
                        mask = torch.zeros_like(original_edge_mask)
                        for h in range(1, hop + 1):
                            if h in edges_by_hop:
                                for edge_idx in edges_by_hop[h]:
                                    mask[edge_idx] = True
                        edge_masks_by_hop[hop] = mask
                
                return edges_by_hop, edge_masks_by_hop, subg_size, ori_mask
        
        except Exception as e:
            print(f"cuGraph processing error for node {vt}: {e}. Falling back to CPU implementation.")
    
    # 如果cuGraph不可用或失败，使用原始实现
    try:
        node_idx, edge_index_sub, _, original_edge_mask = torch_geometric.utils.k_hop_subgraph(
            vt, L, edge_index, relabel_nodes=False
        )
    except RuntimeError:  # Handle memory errors
        print(f"Memory error for node {vt}. Skipping...")
        return None, None, 0, None
        
    if original_edge_mask.sum() > max_subgraph_size:
        print(f"Subgraph for node {vt} too large: {original_edge_mask.sum()} edges. Skipping...")
        return None, None, 0, None

    ori_mask = original_edge_mask
    selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
    
    # Handle case where only one edge is selected
    if selected_edge_positions.dim() == 0:
        selected_edge_positions = selected_edge_positions.unsqueeze(0)
    
    subg_size = selected_edge_positions.size(0)

    # Build adjacency list for fast BFS (more memory efficient)
    adj_list = defaultdict(set)  # Use set instead of list for faster lookup
    for edge_idx in selected_edge_positions:
        src, dst = edge_index[:, edge_idx]
        src_item, dst_item = src.item(), dst.item()
        adj_list[src_item].add(dst_item)
        adj_list[dst_item].add(src_item)

    # BFS to compute hop distances
    hop_distances = {}  # Use dict instead of defaultdict to save memory
    hop_distances[vt] = 0
    queue = deque([vt])
    visited = {vt}  # Use set for O(1) lookup

    while queue:
        current_node = queue.popleft()
        current_hop = hop_distances[current_node]
        for neighbor in adj_list[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                hop_distances[neighbor] = current_hop + 1
                queue.append(neighbor)

    # Group edges by hop with more efficient implementation
    edges_by_hop = defaultdict(list)
    
    for edge_idx in selected_edge_positions:
        src, dst = edge_index[:, edge_idx]
        src_item, dst_item = src.item(), dst.item()
        src_hop = hop_distances.get(src_item, float('inf'))
        dst_hop = hop_distances.get(dst_item, float('inf'))
        edge_hop = min(src_hop, dst_hop) + 1
        if edge_hop <= L + 1:  # Only include edges within our hop limit
            edges_by_hop[edge_hop].append(edge_idx.item())

    # Create edge masks by hop more efficiently
    edge_masks_by_hop = {}
    for hop in range(1, L + 2):
        if hop in edges_by_hop:
            # Create mask only when needed
            mask = torch.zeros_like(original_edge_mask)
            
            # Include current hop and lower hops
            for h in range(1, hop + 1):
                if h in edges_by_hop:
                    for edge_idx in edges_by_hop[h]:
                        mask[edge_idx] = True
                        
            edge_masks_by_hop[hop] = mask

    return edges_by_hop, edge_masks_by_hop, subg_size, ori_mask


# Wrapper for a single node (with error handling)
def precompute_single_node(args):
    vt, G, L, max_subgraph_size = args
    try:
        result = get_edge_sets_by_hop(vt, G, L, max_subgraph_size)
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask = result
        
        if edges_by_hop is None:  # Skip nodes that were too large
            return vt, None
            
        return vt, {
            'edges_by_hop': edges_by_hop,
            'edge_masks_by_hop': {k: v.cpu() for k, v in edge_masks_by_hop.items()},
            'subg_size': subg_size,
            'ori_mask': ori_mask.cpu() if ori_mask is not None else None
        }
    except Exception as e:
        print(f"Error processing node {vt}: {str(e)}")
        return vt, None


# Batch precompute and save with memory management
def precompute_in_batches(G, list_of_nodes, L, num_workers=4, batch_size=50, 
                         save_dir='precomputed/', max_subgraph_size=5000):
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out nodes with too many neighbors if possible
    if hasattr(G, 'num_nodes'):
        degree = torch_geometric.utils.degree(G.edge_index[0], num_nodes=G.num_nodes)
        filtered_nodes = [node for node in list_of_nodes if degree[node] <= max_subgraph_size]
        print(f"Filtered out {len(list_of_nodes) - len(filtered_nodes)} nodes with degree > {max_subgraph_size}")
        list_of_nodes = filtered_nodes
    
    batches = [list_of_nodes[i:i+batch_size] for i in range(0, len(list_of_nodes), batch_size)]
    
    for batch_idx, batch_nodes in enumerate(tqdm(batches, desc="Precomputing batches")):
        batch_results = {}
        args = [(vt, G, L, max_subgraph_size) for vt in batch_nodes]

        with multiprocessing.Pool(num_workers) as pool:
            for vt, result in pool.map(precompute_single_node, args):
                if result is not None:  # Skip failed nodes
                    batch_results[vt] = result

        # Save this batch if not empty
        if batch_results:
            save_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
            torch.save(batch_results, save_path)
            # Clear memory
            del batch_results
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Load batches with memory efficiency
def load_precomputed(save_dir='precomputed/', batch_size=10):
    """Load precomputed data in smaller batches to avoid memory issues"""
    all_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pt')])
    precomputed_data = {}
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_data = {}
        
        for fname in batch_files:
            file_data = torch.load(os.path.join(save_dir, fname))
            batch_data.update(file_data)
            
        precomputed_data.update(batch_data)
        del batch_data  # Free memory
        
    return precomputed_data


# 修改主执行代码，添加GPU检测
if __name__ == "__main__":
    # 检查GPU是否可用并打印状态信息
    if CUGRAPH_AVAILABLE:
        print("Using cuGraph acceleration on GPU")
    else:
        print("GPU acceleration not available, using CPU implementation")
    
    # For BAHouse dataset, use smaller batch sizes and fewer workers
    center_nodes = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))
    max_subgraph_size = 30  # Skip nodes with too many neighbors
    num_workers = 8  # Reduce number of parallel processes 
    batch_size = 50  # Smaller batch size for lower memory usage
    save_dir = './precomputed/{}'.format(data_name)

    precompute_in_batches(data, center_nodes, L, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        save_dir=save_dir,
                        max_subgraph_size=max_subgraph_size)

    # Example usage for loading
    # precomputed_data = load_precomputed(save_dir, batch_size=5)
    # print(f"Loaded data for {len(precomputed_data)} nodes")
