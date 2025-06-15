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

# 创建进程间共享计数器和锁，用于节点级别进度报告
node_counter = multiprocessing.Value('i', 0)
counter_lock = multiprocessing.Lock()


def get_edge_sets_by_hop(vt, G, L, max_subgraph_size=50000):
    edge_index = G.edge_index
    
    # 快速检查节点度数 - 用于提前终止
    node_degree = torch_geometric.utils.degree(edge_index[0], num_nodes=G.num_nodes)
    if node_degree[vt] > max_subgraph_size:
        return None, None, 0, None  # 跳过连接过多的节点
    
    try:
        # 优化: 直接使用PyG的k_hop_subgraph但设置更严格的内存限制
        start_time = time.time()
        node_idx, edge_index_sub, _, original_edge_mask = torch_geometric.utils.k_hop_subgraph(
            vt, L, edge_index, relabel_nodes=False, num_nodes=G.num_nodes
        )
        
        # 子图太大则跳过
        if original_edge_mask.sum() > max_subgraph_size:
            return None, None, 0, None
        
        ori_mask = original_edge_mask
        selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=True)[0]  # 更快的实现
        subg_size = selected_edge_positions.size(0)
        
        # 优化: 使用稀疏矩阵加速BFS
        row, col = edge_index[:, original_edge_mask]
        
        # 更高效的邻接表构建
        adj_list = defaultdict(list)
        for i in range(len(row)):
            src, dst = row[i].item(), col[i].item()
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        # 快速BFS
        hop_distances = {vt: 0}
        queue = deque([vt])
        
        while queue:
            node = queue.popleft()
            current_hop = hop_distances[node]
            
            if current_hop >= L:  # 达到最大跳数时停止
                continue
                
            for neighbor in adj_list[node]:
                if neighbor not in hop_distances:
                    hop_distances[neighbor] = current_hop + 1
                    queue.append(neighbor)
        
        # 优化: 批量分配边到hops
        edges_by_hop = defaultdict(list)
        
        for i in range(len(selected_edge_positions)):
            edge_idx = selected_edge_positions[i].item()
            src, dst = edge_index[:, edge_idx]
            src_hop = hop_distances.get(src.item(), float('inf'))
            dst_hop = hop_distances.get(dst.item(), float('inf'))
            min_hop = min(src_hop, dst_hop)
            
            if min_hop < float('inf'):
                edge_hop = min_hop + 1
                if edge_hop <= L + 1:
                    edges_by_hop[edge_hop].append(edge_idx)
        
        # 优化: 创建边掩码的高效实现
        edge_masks_by_hop = {}
        for hop in range(1, L + 2):
            if hop not in edges_by_hop:
                continue
                
            # 收集当前hop及以下的所有边
            all_edges = []
            for h in range(1, hop + 1):
                if h in edges_by_hop:
                    all_edges.extend(edges_by_hop[h])
            
            # 创建掩码 (向量化操作)
            mask = torch.zeros_like(original_edge_mask)
            if all_edges:  # 只在有边时设置掩码
                mask[all_edges] = True
            edge_masks_by_hop[hop] = mask
        
        # 清理以节省内存
        del adj_list, hop_distances
        
        return edges_by_hop, edge_masks_by_hop, subg_size, ori_mask
        
    except Exception as e:
        print(f"Error processing node {vt}: {str(e)}")
        return None, None, 0, None


# 改进的单节点预计算包装函数
def precompute_single_node(args):
    global node_counter, counter_lock
    vt, G, L, max_subgraph_size, total_nodes = args
    try:
        result = get_edge_sets_by_hop(vt, G, L, max_subgraph_size)
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask = result
        
        # 更新节点计数器并定期报告进度
        with counter_lock:
            node_counter.value += 1
            current_count = node_counter.value
            # 每处理100个节点或是最后一个节点时报告进度
            if current_count % 100 == 0 or current_count == total_nodes:
                print(f"已处理 {current_count}/{total_nodes} 个节点 ({current_count/total_nodes*100:.1f}%)")
        
        if edges_by_hop is None:  # 跳过太大或处理失败的节点
            print(f"跳过节点 {vt} - 处理失败或子图过大")
            return vt, None
            
        return vt, {
            'edges_by_hop': edges_by_hop,
            'edge_masks_by_hop': {k: v.cpu() for k, v in edge_masks_by_hop.items()},
            'subg_size': subg_size,
            'ori_mask': ori_mask.cpu() if ori_mask is not None else None
        }
    except Exception as e:
        print(f"处理节点 {vt} 时发生意外错误: {str(e)}")
        return vt, None


# 优化的多线程处理（使用更大的批次大小和更少的工作进程）
def precompute_in_batches(G, list_of_nodes, L, num_workers=4, batch_size=100, 
                         save_dir='precomputed/', max_subgraph_size=5000):
    import time
    os.makedirs(save_dir, exist_ok=True)
    
    # 重置节点计数器
    global node_counter
    with counter_lock:
        node_counter.value = 0
        
    # 记录失败的节点
    failed_nodes_file = os.path.join(save_dir, "failed_nodes.txt")
    failed_nodes = []
    
    # 过滤掉度数太大的节点 - 更快的过滤
    print("按度数过滤节点...")
    if hasattr(G, 'num_nodes'):
        degree = torch_geometric.utils.degree(G.edge_index[0], num_nodes=G.num_nodes)
        filtered_nodes = []
        for node in tqdm(list_of_nodes, desc="检查节点度数"):
            if node < len(degree) and degree[node] <= max_subgraph_size:
                filtered_nodes.append(node)
            else:
                failed_nodes.append(node)
        
        print(f"过滤掉 {len(list_of_nodes) - len(filtered_nodes)} 个度数 > {max_subgraph_size} 的节点")
        list_of_nodes = filtered_nodes
    
    # 计算总节点数用于进度报告
    total_nodes = len(list_of_nodes)
    print(f"开始处理 {total_nodes} 个节点...")
    
    # 创建更大的批次以减少开销
    batches = [list_of_nodes[i:i+batch_size] for i in range(0, len(list_of_nodes), batch_size)]
    print(f"处理 {len(batches)} 个批次，每批 {batch_size} 个节点")
    
    total_start_time = time.time()
    
    for batch_idx, batch_nodes in enumerate(tqdm(batches, desc="预计算批次")):
        batch_start_time = time.time()
        batch_results = {}
        args = [(vt, G, L, max_subgraph_size, total_nodes) for vt in batch_nodes]
        
        # 使用进程池处理批次
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(precompute_single_node, args),
                total=len(args),
                desc=f"批次 {batch_idx+1}/{len(batches)}",
                leave=False
            ))
            
            for vt, result in results:
                if result is not None:
                    batch_results[vt] = result
                else:
                    failed_nodes.append(vt)
        
        # 保存这个批次（如果不为空）
        if batch_results:
            save_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
            try:
                torch.save(batch_results, save_path)
                batch_time = time.time() - batch_start_time
                print(f"保存批次 {batch_idx}，包含 {len(batch_results)} 个节点，耗时 {batch_time:.2f}秒")
            except Exception as e:
                print(f"保存批次 {batch_idx} 时出错: {str(e)}")
                # 记录整个批次的失败节点
                failed_nodes.extend(batch_nodes)
        
        # 清理内存
        del batch_results, results
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 每10个批次保存一次进度
        if (batch_idx + 1) % 10 == 0:
            # 保存失败的节点列表
            with open(failed_nodes_file, 'w') as f:
                for node in failed_nodes:
                    f.write(f"{node}\n")
            print(f"进度: {batch_idx+1}/{len(batches)} 批次, 已用时间: {time.time()-total_start_time:.2f}秒")
            print(f"已处理 {node_counter.value}/{total_nodes} 个节点 ({node_counter.value/total_nodes*100:.1f}%)")
    
    # 最终保存失败的节点列表
    with open(failed_nodes_file, 'w') as f:
        for node in failed_nodes:
            f.write(f"{node}\n")
    print(f"已保存 {len(failed_nodes)} 个失败节点到 {failed_nodes_file}")
    print(f"总时间: {time.time()-total_start_time:.2f}秒")
    print(f"最终处理了 {node_counter.value}/{total_nodes} 个节点 ({node_counter.value/total_nodes*100:.1f}%)")

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
    import time
    print("Using optimized PyG implementation for graph processing")
    
    # 加载测试节点
    center_nodes = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))
    print(f"Loaded {len(center_nodes)} center nodes")
    
    # 优化参数
    max_subgraph_size = 500  # 增加阈值以处理更多节点
    num_workers = max(4, os.cpu_count() // 2)  # 使用一半的CPU核心
    batch_size = 200  # 更大的批次减少开销
    save_dir = './precomputed/{}'.format(data_name)
    
    print(f"Starting precomputation with parameters:")
    print(f"- Max subgraph size: {max_subgraph_size}")
    print(f"- Workers: {num_workers}")
    print(f"- Batch size: {batch_size}")
    print(f"- Save directory: {save_dir}")

    start_time = time.time()
    precompute_in_batches(data, center_nodes, L, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        save_dir=save_dir,
                        max_subgraph_size=max_subgraph_size)
    print(f"Precomputation completed in {time.time()-start_time:.2f} seconds")

    # 验证加载过程
    print("Verifying data loading...")
    sample_data = load_precomputed(save_dir, batch_size=5)
    print(f"Verification complete: loaded {len(sample_data)} nodes")
