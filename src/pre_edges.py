# 自适应导入utils模块
try:
    from utils import *  # 当在src目录内运行时
except ImportError:
    try:
        from src.utils import *  # 当在项目根目录运行时
    except ImportError:
        raise ImportError("无法导入utils模块，请确保在正确的目录中运行脚本")
import os

# 确保正确找到配置文件 - 支持从src目录或项目根目录运行
config_path = "config.yaml"
if not os.path.exists(config_path):
    config_path = "../config.yaml"
    
if not os.path.exists(config_path):
    raise FileNotFoundError("找不到config.yaml文件，请确保它在当前目录或上一级目录中")
    
config = load_config(config_path)
data_name = config['data_name']
random_seed = config['random_seed']
L = config['L']
set_seed(random_seed)
data = dataset_func(config)

import torch
import torch.cuda
import torch_geometric
import os
import sys
import multiprocessing
import gc
import numpy as np
import time
from tqdm import tqdm
from collections import deque, defaultdict

# 尝试导入可选模块
try:
    import psutil
except ImportError:
    print("警告: psutil 模块不可用，部分资源检测功能将被禁用")
    psutil = None

try:
    import torch_sparse
except ImportError:
    print("警告: torch_sparse 模块不可用，将使用标准实现")
import gc
import math
from tqdm import tqdm
from collections import deque, defaultdict
from functools import lru_cache

# 智能设置设备 - 自动检测并使用可用GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 硬件资源检测 - 增强健壮性
NUM_CPUS = os.cpu_count() or 4  # 默认为4核心
CPU_MEMORY = 0

# 安全获取系统内存
try:
    if hasattr(psutil, 'virtual_memory'):
        CPU_MEMORY = psutil.virtual_memory().total / (1024 ** 3)  # GB
    elif sys.platform == 'darwin':  # macOS
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            CPU_MEMORY = int(result.stdout) / (1024 ** 3)
    elif sys.platform == 'linux':  # Linux
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    CPU_MEMORY = int(line.split()[1]) / (1024 ** 2)  # KB to GB
                    break
except Exception as e:
    print(f"警告: 无法获取系统内存信息: {str(e)}")
    CPU_MEMORY = 8  # 假设8GB内存

# 安全获取GPU内存
GPU_MEMORY = 0
if torch.cuda.is_available():
    try:
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    except Exception as e:
        print(f"警告: 无法获取GPU内存信息: {str(e)}")

# 打印系统资源信息
print(f"系统资源: {NUM_CPUS} CPU核心", end="")
if CPU_MEMORY > 0:
    print(f", {CPU_MEMORY:.1f}GB CPU内存", end="")
if GPU_MEMORY > 0:
    print(f", {GPU_MEMORY:.1f}GB GPU内存", end="")
print()

# Set multiprocessing method early (important for MacOS/Linux)
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    print("警告: 已经设置了multiprocessing方法，继续使用现有设置")

# 创建进程间共享计数器和锁，用于节点级别进度报告
node_counter = multiprocessing.Value('i', 0)
counter_lock = multiprocessing.Lock()

# 智能设置设备 - 自动检测并使用可用GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 硬件资源检测 - 避免重复检测内存
NUM_CPUS = os.cpu_count()
# 不再重复检测CPU内存，使用前面已经检测到的值
    
GPU_MEMORY = 0
if torch.cuda.is_available():
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB

print(f"系统资源: {NUM_CPUS} CPU核心", end="")
if CPU_MEMORY > 0:
    print(f", {CPU_MEMORY:.1f}GB CPU内存", end="")
if GPU_MEMORY > 0:
    print(f", {GPU_MEMORY:.1f}GB GPU内存", end="")
print()

# 全局配置参数 - 将根据硬件自动调整
AUTO_CONFIG = {
    'use_gpu': torch.cuda.is_available(),
    'sparse_matrix': True,  # 使用稀疏矩阵表示
    'vectorized_ops': True,  # 使用向量化操作
    'adaptive_batching': True,  # 根据节点度数自适应批次大小
    'memory_limit_gb': min(CPU_MEMORY * 0.7, 32),  # 使用70%的系统内存或最多32GB
    'checkpoint_interval': 10  # 每处理10个批次保存一次检查点
}


def get_edge_sets_by_hop(vt, G, L, max_subgraph_size=50000, use_gpu=False):
    """
    获取从节点vt出发的L跳内所有边的分层集合
    采用极速优化算法：混合CPU/GPU计算、稀疏矩阵、向量化操作
    
    Args:
        vt: 中心节点ID
        G: PyG图对象
        L: 最大跳数
        max_subgraph_size: 子图最大大小限制
        use_gpu: 是否使用GPU加速
    
    Returns:
        edges_by_hop: 按跳数分类的边索引字典
        edge_masks_by_hop: 按跳数累积的边掩码字典
        subg_size: 子图大小
        ori_mask: 原始边掩码
    """
    # 提前检测设备，按需转移数据以减少内存复制
    device = DEVICE if use_gpu and AUTO_CONFIG['use_gpu'] else torch.device('cpu')
    edge_index = G.edge_index
    
    # 快速检查节点度数 - 用于提前终止
    node_degree = torch_geometric.utils.degree(edge_index[0], num_nodes=G.num_nodes)
    if node_degree[vt] > max_subgraph_size:
        return None, None, 0, None  # 跳过连接过多的节点
    
    try:
        # 对超大度节点采用采样策略而非完整处理
        if node_degree[vt] > max_subgraph_size * 0.5:
            sampling_ratio = max(0.2, max_subgraph_size / (node_degree[vt] * 2))
            print(f"节点 {vt} 度数很大 ({node_degree[vt]}), 采用 {sampling_ratio:.2f} 采样率")
            return sample_large_degree_node(vt, G, L, sampling_ratio, max_subgraph_size)
        
        # 优化: 获取L跳子图
        start_time = time.time()
        node_idx, edge_index_sub, _, original_edge_mask = torch_geometric.utils.k_hop_subgraph(
            vt, L, edge_index, relabel_nodes=False, num_nodes=G.num_nodes
        )
        
        # 子图太大则跳过
        if original_edge_mask.sum() > max_subgraph_size:
            print(f"节点 {vt} 子图过大 ({original_edge_mask.sum()} > {max_subgraph_size})")
            return None, None, 0, None
        
        # 记录原始掩码并获取边索引位置
        ori_mask = original_edge_mask
        selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=True)[0]
        subg_size = selected_edge_positions.size(0)
        
        # 使用CSR稀疏矩阵表示子图
        edge_index_subset = edge_index[:, original_edge_mask]
        
        # 根据是否使用GPU选择实现路径
        if device.type == 'cuda' and subg_size > 1000:
            # GPU加速版本的BFS和边分类
            return gpu_edge_classification(vt, edge_index_subset, original_edge_mask, L, device)
        else:
            # CPU版本实现 - 采用稀疏表示
            return cpu_edge_classification(vt, edge_index_subset, original_edge_mask, L, selected_edge_positions)
    
    except Exception as e:
        print(f"处理节点 {vt} 时出错: {str(e)}")
        return None, None, 0, None


# 改进的单节点预计算包装函数
def sample_large_degree_node(vt, G, L, sampling_ratio, max_subgraph_size):
    """
    对超大度节点采用采样策略，避免处理完整邻居集
    
    Args:
        vt: 中心节点ID
        G: PyG图对象
        L: 最大跳数
        sampling_ratio: 采样比例
        max_subgraph_size: 最大子图大小
    
    Returns:
        与get_edge_sets_by_hop函数相同的返回值
    """
    edge_index = G.edge_index
    
    # 1. 获取与节点相连的所有边
    node_mask = (edge_index[0] == vt) | (edge_index[1] == vt)
    connected_edges = torch.nonzero(node_mask).squeeze()
    
    # 2. 随机采样边
    num_edges = connected_edges.size(0)
    sample_size = int(num_edges * sampling_ratio)
    perm = torch.randperm(num_edges)[:sample_size]
    sampled_edges = connected_edges[perm]
    
    # 3. 构建采样子图
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    edge_mask[sampled_edges] = True
    
    # 4. 从采样子图出发获取L跳子图
    neighbors = torch.unique(edge_index[:, edge_mask].reshape(-1))
    
    # 5. 从采样后的邻居开始获取L-1跳子图
    node_idx, edge_index_sub, _, original_edge_mask = torch_geometric.utils.k_hop_subgraph(
        neighbors, L-1, edge_index, relabel_nodes=False, num_nodes=G.num_nodes
    )
    
    # 如果子图仍然太大，进一步减少采样率
    if original_edge_mask.sum() > max_subgraph_size:
        return sample_large_degree_node(vt, G, L, sampling_ratio * 0.5, max_subgraph_size)
    
    # 6. 使用标准方法处理采样后的子图
    row, col = edge_index[:, original_edge_mask]
    selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=True)[0]
    
    # 构建稀疏邻接表
    adj_list = {}
    for i in range(len(row)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        if dst not in adj_list:
            adj_list[dst] = []
        adj_list[src].append(dst)
        adj_list[dst].append(src)
    
    # 使用NumPy数组存储距离
    max_node_id = max(max(adj_list.keys()), vt) + 1
    hop_distances = np.full(max_node_id, np.inf, dtype=np.float32)
    hop_distances[vt] = 0
    
    # BFS实现
    queue = deque([vt])
    visited = {vt}
    
    while queue:
        node = queue.popleft()
        current_hop = hop_distances[node]
        
        if current_hop >= L:
            continue
        
        neighbors = adj_list.get(node, [])
        for neighbor in neighbors:
            if hop_distances[neighbor] == np.inf:
                hop_distances[neighbor] = current_hop + 1
                queue.append(neighbor)
                visited.add(neighbor)
    
    # 边分类
    edges_by_hop = defaultdict(list)
    
    # 批量处理
    batch_size = 1000
    for start_idx in range(0, len(selected_edge_positions), batch_size):
        end_idx = min(start_idx + batch_size, len(selected_edge_positions))
        batch_indices = selected_edge_positions[start_idx:end_idx]
        
        batch_edges = edge_index[:, batch_indices]
        src_nodes = batch_edges[0].numpy()
        dst_nodes = batch_edges[1].numpy()
        
        src_hops = hop_distances[src_nodes]
        dst_hops = hop_distances[dst_nodes]
        min_hops = np.minimum(src_hops, dst_hops)
        
        for i, (edge_idx, min_hop) in enumerate(zip(batch_indices, min_hops)):
            if min_hop < float('inf'):
                edge_hop = min_hop + 1
                if edge_hop <= L + 1:
                    edges_by_hop[int(edge_hop)].append(edge_idx.item())
    
    # 创建边掩码
    edge_masks_by_hop = {}
    for hop in range(1, L + 2):
        if hop not in edges_by_hop:
            continue
            
        all_edges = [edge for h in range(1, hop + 1) for edge in edges_by_hop.get(h, [])]
        
        if all_edges:
            mask = torch.zeros_like(original_edge_mask)
            mask[all_edges] = True
            edge_masks_by_hop[hop] = mask
        else:
            edge_masks_by_hop[hop] = torch.zeros_like(original_edge_mask)
    
    # 清理内存
    del adj_list, hop_distances, visited
    
    return edges_by_hop, edge_masks_by_hop, original_edge_mask.sum().item(), original_edge_mask


def cpu_edge_classification(vt, edge_index_subset, original_edge_mask, L, selected_edge_positions):
    """
    CPU版本的边分类实现，针对边数更少的情况优化
    
    Args:
        vt: 中心节点
        edge_index_subset: 子图边索引
        original_edge_mask: 原始边掩码
        L: 最大跳数
        selected_edge_positions: 被选中的边位置
    
    Returns:
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask
    """
    row, col = edge_index_subset[0], edge_index_subset[1]
    
    # 构建稀疏邻接表
    adj_list = {}
    for i in range(len(row)):
        src, dst = row[i].item(), col[i].item()
        if src not in adj_list:
            adj_list[src] = []
        if dst not in adj_list:
            adj_list[dst] = []
        adj_list[src].append(dst)
        adj_list[dst].append(src)
    
    # 使用NumPy数组存储距离
    max_node_id = max(max(adj_list.keys()), vt) + 1
    hop_distances = np.full(max_node_id, np.inf, dtype=np.float32)
    hop_distances[vt] = 0
    
    # 优化的BFS实现
    queue = deque([vt])
    visited = {vt}  # 使用集合跟踪已访问节点，避免重复
    
    while queue:
        node = queue.popleft()
        current_hop = hop_distances[node]
        
        if current_hop >= L:  # 达到最大跳数时停止
            continue
        
        # 批量处理邻居
        neighbors = adj_list.get(node, [])
        for neighbor in neighbors:
            if hop_distances[neighbor] == np.inf:  # 未访问过
                hop_distances[neighbor] = current_hop + 1
                queue.append(neighbor)
                visited.add(neighbor)
    
    # 向量化边分类 - 预分配数组并批量处理
    edges_by_hop = defaultdict(list)
    
    # 批量获取源节点和目标节点的跳数
    batch_size = 5000  # 增大批次以提高效率
    for start_idx in range(0, len(selected_edge_positions), batch_size):
        end_idx = min(start_idx + batch_size, len(selected_edge_positions))
        batch_indices = selected_edge_positions[start_idx:end_idx]
        
        # 获取批次中的边
        batch_edges = edge_index_subset[:, start_idx:min(end_idx, edge_index_subset.size(1)-1)]
        src_nodes = batch_edges[0].numpy()
        dst_nodes = batch_edges[1].numpy()
        
        # 向量化获取跳数
        src_hops = hop_distances[src_nodes]
        dst_hops = hop_distances[dst_nodes]
        min_hops = np.minimum(src_hops, dst_hops)
        
        # 向量化边分类
        edge_hops = min_hops + 1
        valid_mask = ~np.isinf(min_hops) & (edge_hops <= L + 1)
        
        # 按跳数分类边
        for i, (edge_idx, hop, valid) in enumerate(zip(batch_indices, edge_hops, valid_mask)):
            if valid:
                edges_by_hop[int(hop)].append(edge_idx.item())
    
    # 高效创建边掩码
    edge_masks_by_hop = {}
    for hop in range(1, L + 2):
        if hop not in edges_by_hop:
            continue
            
        # 收集当前hop及以下的所有边 - 使用列表推导更快
        all_edges = [edge for h in range(1, hop + 1) for edge in edges_by_hop.get(h, [])]
        
        # 创建掩码
        if all_edges:
            mask = torch.zeros_like(original_edge_mask)
            mask[all_edges] = True
            edge_masks_by_hop[hop] = mask
        else:
            edge_masks_by_hop[hop] = torch.zeros_like(original_edge_mask)
    
    # 清理内存
    del adj_list, hop_distances, visited
    
    return edges_by_hop, edge_masks_by_hop, original_edge_mask.sum().item(), original_edge_mask


def gpu_edge_classification(vt, edge_index_subset, original_edge_mask, L, device):
    """
    GPU加速的边分类实现
    
    Args:
        vt: 中心节点
        edge_index_subset: 子图边索引
        original_edge_mask: 原始边掩码
        L: 最大跳数
        device: GPU设备
    
    Returns:
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask
    """
    try:
        # 将数据移至GPU
        edge_index_gpu = edge_index_subset.to(device)
        
        # 1. 构建邻接表表示
        num_nodes = edge_index_gpu.max().item() + 1
        row, col = edge_index_gpu[0], edge_index_gpu[1]
        
        # 构建邻接矩阵 (GPU加速)
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
        adj_matrix[row, col] = True  # 设置边
        
        # 2. 在GPU上执行BFS
        distances = torch.full((num_nodes,), float('inf'), device=device)
        distances[vt] = 0
        
        # 迭代式BFS
        for hop in range(L):
            # 找出当前层的节点
            current_nodes = (distances == hop).nonzero().squeeze(1)
            if len(current_nodes) == 0:
                break
                
            # 获取邻居
            neighbors = adj_matrix[current_nodes].nonzero()
            if len(neighbors) == 0:
                continue
                
            neighbor_rows = neighbors[:, 0]  # 对应current_nodes的索引
            neighbor_cols = neighbors[:, 1]  # 邻居节点ID
            
            # 更新距离 (只更新未访问过的节点)
            unvisited = torch.isinf(distances[neighbor_cols])
            distances[neighbor_cols[unvisited]] = hop + 1
        
        # 3. 边分类
        edges_by_hop = defaultdict(list)
        
        # 将边索引移至GPU
        edge_indices = torch.arange(original_edge_mask.size(0))[original_edge_mask].to(device)
        
        # 计算所有边的跳数 (批量处理)
        src_hops = distances[row]
        dst_hops = distances[col]
        min_hops = torch.minimum(src_hops, dst_hops)
        edge_hops = min_hops + 1
        
        # 有效边 (跳数有限且不超过L+1)
        valid_mask = ~torch.isinf(min_hops) & (edge_hops <= L + 1)
        
        # 将边分组到对应的跳数
        for h in range(1, L + 2):
            hop_mask = valid_mask & (edge_hops == h)
            if hop_mask.any():
                # 获取此跳数的边索引
                edges_in_hop = edge_indices[hop_mask].cpu().tolist()
                edges_by_hop[h] = edges_in_hop
        
        # 4. 创建边掩码
        edge_masks_by_hop = {}
        for hop in range(1, L + 2):
            if hop not in edges_by_hop:
                continue
            
            # 合并当前跳数及以下的所有边
            all_edges = []
            for h in range(1, hop + 1):
                if h in edges_by_hop:
                    all_edges.extend(edges_by_hop[h])
            
            # 创建掩码
            if all_edges:
                mask = torch.zeros_like(original_edge_mask)
                mask[all_edges] = True
                edge_masks_by_hop[hop] = mask
            else:
                edge_masks_by_hop[hop] = torch.zeros_like(original_edge_mask)
        
        # 清理 GPU 内存
        del adj_matrix, distances, edge_index_gpu
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return edges_by_hop, edge_masks_by_hop, original_edge_mask.sum().item(), original_edge_mask
        
    except Exception as e:
        print(f"GPU处理失败，回退到CPU: {str(e)}")
        # 回退到CPU处理
        selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=True)[0]
        return cpu_edge_classification(vt, edge_index_subset, original_edge_mask, L, selected_edge_positions)


def precompute_single_node(args):
    global node_counter, counter_lock
    vt, G, L, max_subgraph_size, total_nodes, use_gpu = args
    try:
        # 配置节点处理参数 - 根据节点度数自适应调整
        node_degree = torch_geometric.utils.degree(G.edge_index[0], num_nodes=G.num_nodes)[vt].item()
        
        # 对高度节点使用GPU加速（如果可用）
        node_use_gpu = use_gpu and node_degree > 1000
        
        # 处理节点
        result = get_edge_sets_by_hop(vt, G, L, max_subgraph_size, node_use_gpu)
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask = result
        
        # 更新节点计数器并定期报告进度
        with counter_lock:
            node_counter.value += 1
            current_count = node_counter.value
            # 每处理10个节点或是最后一个节点时报告进度
            if current_count % 10 == 0 or current_count == total_nodes:
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


# 智能批次大小选择函数
def estimate_optimal_batch_size(node_degrees, max_memory_gb=16):
    """根据节点度数分布智能估算最佳批次大小"""
    # 根据系统内存自动调整
    system_memory_gb = max_memory_gb  # 默认值
    
    # 尝试获取实际系统内存
    try:
        system_memory_gb = max_memory_gb  # 默认值
        if CPU_MEMORY > 0:  # 使用全局已检测到的内存
            system_memory_gb = CPU_MEMORY
        elif hasattr(psutil, 'virtual_memory') and psutil is not None:
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass  # 使用默认内存估计
        
    # 安全系数 - 只使用70%的可用内存
    max_memory_gb = min(max_memory_gb, system_memory_gb * 0.7)
    
    # 计算平均节点度数和标准差
    avg_degree = np.mean(node_degrees)
    std_degree = np.std(node_degrees)
    
    # 估算每个节点的平均内存消耗 (GB)
    # 假设每个边需要约100字节的内存 (包括各种数据结构开销)
    memory_per_edge_gb = 100 / (1024**3)
    avg_memory_per_node_gb = avg_degree * memory_per_edge_gb * 5  # *5作为安全系数
    
    # 调整系数 - 如果度数变化大，则更保守
    if std_degree > avg_degree * 2:
        safety_factor = 2.0
    else:
        safety_factor = 1.0
        
    # 计算批次大小 (至少5个节点，最多500个)
    batch_size = int(max(5, min(500, max_memory_gb / (avg_memory_per_node_gb * safety_factor))))
    
    print(f"节点度数统计: 平均={avg_degree:.1f}, 标准差={std_degree:.1f}")
    print(f"估计每节点内存: {avg_memory_per_node_gb*1024:.2f}MB, 安全系数: {safety_factor}")
    
    return batch_size


# 高效的多级并行处理框架
def precompute_in_batches(G, list_of_nodes, L, num_workers=4, batch_size=100, 
                         save_dir='precomputed/', max_subgraph_size=5000):
    """
    优化的多级并行处理框架，支持自适应批处理、检查点恢复和混合CPU/GPU计算
    
    Args:
        G: PyG图对象
        list_of_nodes: 要处理的节点列表
        L: 最大跳数
        num_workers: 工作进程数
        batch_size: 批次大小
        save_dir: 保存目录
        max_subgraph_size: 最大子图大小
    """
    import time, json
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查点恢复逻辑 - 检查是否有之前的进度
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")
    processed_nodes = set()
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_nodes = set(checkpoint.get('processed_nodes', []))
                print(f"检查点恢复: 已找到 {len(processed_nodes)} 个已处理节点的记录")
        except Exception as e:
            print(f"读取检查点失败: {str(e)}")
    
    # 重置节点计数器
    global node_counter
    with counter_lock:
        node_counter.value = 0
        
    # 记录失败的节点
    failed_nodes_file = os.path.join(save_dir, "failed_nodes.txt")
    failed_nodes = []
    
    # 过滤已处理的节点
    if processed_nodes:
        original_count = len(list_of_nodes)
        list_of_nodes = [node for node in list_of_nodes if node not in processed_nodes]
        print(f"跳过 {original_count - len(list_of_nodes)} 个已处理的节点")
    
    # 过滤掉度数太大的节点 - 使用向量化操作加速
    print("按度数过滤节点...")
    if hasattr(G, 'num_nodes'):
        degree = torch_geometric.utils.degree(G.edge_index[0], num_nodes=G.num_nodes)
        degree_np = degree.numpy()  # 转换为NumPy数组以加速过滤
        
        # 创建过滤掩码
        filtered_nodes = []
        skipped_nodes = []
        
        # 使用向量化操作更快地过滤
        for node in tqdm(list_of_nodes, desc="检查节点度数"):
            if node < len(degree_np) and degree_np[node] <= max_subgraph_size:
                filtered_nodes.append(node)
            else:
                skipped_nodes.append(node)
        
        print(f"过滤掉 {len(skipped_nodes)} 个度数 > {max_subgraph_size} 的节点")
        failed_nodes.extend(skipped_nodes)
        list_of_nodes = filtered_nodes
    
    # 计算总节点数用于进度报告
    total_nodes = len(list_of_nodes)
    if total_nodes == 0:
        print("没有需要处理的节点，完成!")
        return
        
    print(f"开始处理 {total_nodes} 个节点...")
    
    # 按节点度数对节点进行分层 - 更高效的处理顺序
    print("按度数对节点分层...")
    node_degrees = []
    for node in list_of_nodes:
        if node < len(degree):
            node_degrees.append(degree[node].item())
        else:
            node_degrees.append(0)
            
    # 智能批次大小计算
    try:
        estimated_batch_size = estimate_optimal_batch_size(node_degrees, max_memory_gb=16)
        print(f"自适应批次大小: {estimated_batch_size} (原始: {batch_size})")
        batch_size = estimated_batch_size
    except Exception as e:
        print(f"使用默认批次大小: {batch_size}, 原因: {str(e)}")
    
    # 创建更智能的批次 - 将相似度数的节点分到一起
    nodes_by_degree = defaultdict(list)
    for node, node_degree in zip(list_of_nodes, node_degrees):
        # 分桶: <100, 100-500, 500-1000, 1000-5000, >5000
        if node_degree < 100:
            bucket = 'tiny'
        elif node_degree < 500:
            bucket = 'small'
        elif node_degree < 1000:
            bucket = 'medium'
        elif node_degree < max_subgraph_size:
            bucket = 'large'
        else:
            bucket = 'huge'
        nodes_by_degree[bucket].append(node)
    
    # 处理顺序: tiny -> small -> medium -> large -> huge
    processing_order = ['tiny', 'small', 'medium', 'large', 'huge']
    ordered_nodes = []
    for bucket in processing_order:
        if bucket in nodes_by_degree:
            bucket_nodes = nodes_by_degree[bucket]
            print(f"{bucket} 桶: {len(bucket_nodes)} 个节点")
            ordered_nodes.extend(bucket_nodes)
    
    # 创建批次 - 保持相似度数的节点在一起
    batches = [ordered_nodes[i:i+batch_size] for i in range(0, len(ordered_nodes), batch_size)]
    print(f"处理 {len(batches)} 个批次，每批约 {batch_size} 个节点")
    
    # 自动调整worker数量
    recommended_workers = min(num_workers, os.cpu_count() - 1, 8)  # 保留至少一个核心给系统
    if recommended_workers != num_workers:
        print(f"自动调整worker数量: {num_workers} -> {recommended_workers}")
        num_workers = recommended_workers
    
    total_start_time = time.time()
    
    # 检测GPU是否可用
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU加速: 启用 ({gpu_info.name}, {gpu_info.total_memory/1024**3:.1f}GB)")
    else:
        print("GPU加速: 禁用 (未检测到CUDA设备)")
    
    # 进度跟踪
    total_processed = len(processed_nodes)  # 包括已有的处理节点
    checkpoint_data = {'processed_nodes': list(processed_nodes)}
    
    for batch_idx, batch_nodes in enumerate(tqdm(batches, desc="预计算批次")):
        batch_start_time = time.time()
        batch_results = {}
        
        # 准备参数 (包括GPU标志)
        args = [(vt, G, L, max_subgraph_size, total_nodes, use_gpu) for vt in batch_nodes]
        
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
                    processed_nodes.add(vt)
                else:
                    failed_nodes.append(vt)
        
        # 保存这个批次（如果不为空）
        if batch_results:
            save_path = os.path.join(save_dir, f'batch_{batch_idx}.pt')
            try:
                torch.save(batch_results, save_path)
                batch_time = time.time() - batch_start_time
                total_processed += len(batch_results)
                print(f"保存批次 {batch_idx}，包含 {len(batch_results)} 个节点，耗时 {batch_time:.2f}秒")
            except Exception as e:
                print(f"保存批次 {batch_idx} 时出错: {str(e)}")
                # 记录整个批次的失败节点
                failed_nodes.extend(batch_nodes)
        
        # 清理内存
        del batch_results, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # 强制垃圾回收
        
        # 更新并保存检查点
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
            # 更新检查点
            checkpoint_data['processed_nodes'] = list(processed_nodes)
            checkpoint_data['last_batch'] = batch_idx
            checkpoint_data['timestamp'] = time.time()
            
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"检查点已保存: {len(processed_nodes)} 个已处理节点")
            except Exception as e:
                print(f"保存检查点失败: {str(e)}")
            
            # 保存失败的节点列表
            with open(failed_nodes_file, 'w') as f:
                for node in failed_nodes:
                    f.write(f"{node}\n")
            
            # 报告进度
            elapsed = time.time() - total_start_time
            remaining = (elapsed / (batch_idx + 1)) * (len(batches) - batch_idx - 1)
            print(f"进度: {batch_idx+1}/{len(batches)} 批次, 已用时间: {elapsed:.2f}秒, 预计剩余: {remaining:.2f}秒")
            print(f"已处理 {total_processed}/{total_nodes} 个节点 ({total_processed/total_nodes*100:.1f}%)")
            
            # 报告内存使用情况
            try:
                if hasattr(psutil, 'Process'):
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    print(f"内存使用: {memory_info.rss/1024**3:.2f}GB")
            except Exception:
                pass  # 如果无法获取内存信息，则跳过报告
    
    # 最终保存失败的节点列表
    with open(failed_nodes_file, 'w') as f:
        for node in failed_nodes:
            f.write(f"{node}\n")
    
    # 报告最终统计信息
    success_rate = total_processed / (total_nodes + len(processed_nodes)) * 100 if (total_nodes + len(processed_nodes)) > 0 else 0
    print(f"已保存 {len(failed_nodes)} 个失败节点到 {failed_nodes_file}")
    print(f"总时间: {time.time()-total_start_time:.2f}秒")
    print(f"最终处理了 {total_processed}/{total_nodes+len(processed_nodes)} 个节点 (成功率: {success_rate:.1f}%)")

# 惰性加载器类 - 按需加载数据
class LazyLoadPrecomputed:
    """惰性加载预计算数据的类，节省内存"""
    def __init__(self, save_dir, file_list=None):
        self.save_dir = save_dir
        if file_list is None:
            self.file_list = sorted([f for f in os.listdir(save_dir) if f.endswith('.pt')])
        else:
            self.file_list = file_list
        self.file_map = {}  # 映射节点到文件
        self.cache = {}  # 节点缓存
        self.cache_hits = 0
        self.cache_misses = 0
        self.build_node_index()
        
    def build_node_index(self):
        """构建节点到文件的索引映射"""
        print("构建节点索引映射...")
        for fname in tqdm(self.file_list, desc="索引预计算文件"):
            file_path = os.path.join(self.save_dir, fname)
            try:
                # 只加载文件的键，不加载值
                data = torch.load(file_path, map_location='cpu')
                for node in data.keys():
                    self.file_map[node] = fname
            except Exception as e:
                print(f"索引文件 {fname} 时出错: {str(e)}")
        print(f"索引构建完成: 找到 {len(self.file_map)} 个节点")
    
    def __getitem__(self, node):
        """按需加载节点数据"""
        # 检查缓存
        if node in self.cache:
            self.cache_hits += 1
            return self.cache[node]
        
        # 缓存未命中，从文件加载
        self.cache_misses += 1
        if node not in self.file_map:
            raise KeyError(f"节点 {node} 不在预计算数据中")
            
        fname = self.file_map[node]
        file_path = os.path.join(self.save_dir, fname)
        
        try:
            data = torch.load(file_path, map_location='cpu')
            # 更新缓存 (只缓存请求的节点)
            self.cache[node] = data[node]
            
            # 缓存大小控制 - 保持在1000个节点以内
            if len(self.cache) > 1000:
                # 删除最早加入的20%
                nodes_to_remove = list(self.cache.keys())[:200]
                for old_node in nodes_to_remove:
                    del self.cache[old_node]
                    
            return self.cache[node]
        except Exception as e:
            print(f"加载节点 {node} 数据时出错: {str(e)}")
            raise
    
    def __contains__(self, node):
        """检查节点是否存在于预计算数据中"""
        return node in self.file_map
        
    def __len__(self):
        """返回预计算数据中的节点总数"""
        return len(self.file_map)
    
    def report_stats(self):
        """报告缓存命中率等统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
        print(f"缓存统计: 命中 {self.cache_hits}, 未命中 {self.cache_misses}, 命中率 {hit_rate:.1f}%")
        print(f"当前缓存大小: {len(self.cache)} 个节点")


# 增强的加载函数，支持容错和内存优化
def load_precomputed(save_dir='precomputed/', batch_size=10, lazy_loading=False):
    """
    加载预计算数据，带错误处理、日志和内存优化
    
    Args:
        save_dir: 保存目录
        batch_size: 加载批次大小
        lazy_loading: 是否使用惰性加载 (仅在需要时加载)
    
    Returns:
        预计算数据字典 (或惰性加载器)
    """
    all_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pt')])
    failed_files = []
    
    print(f"加载 {len(all_files)} 个批次文件，来自 {save_dir}")
    
    # 惰性加载实现 - 创建按需加载的接口
    if lazy_loading:
        return LazyLoadPrecomputed(save_dir, all_files)
    
    # 标准加载实现
    precomputed_data = {}
    memory_usage_start = 0
    
    # 尝试获取初始内存使用
    try:
        if hasattr(psutil, 'Process'):
            memory_usage_start = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass  # 如果获取内存信息失败，忽略
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_start_time = time.time()
        
        for fname in batch_files:
            try:
                file_path = os.path.join(save_dir, fname)
                file_data = torch.load(file_path)
                precomputed_data.update(file_data)
            except Exception as e:
                print(f"加载文件 {fname} 时出错: {str(e)}")
                failed_files.append(fname)
        
        # 周期性报告进度
        batch_time = time.time() - batch_start_time
        if (i//batch_size) % 5 == 0 or i + batch_size >= len(all_files):
            memory_info = ""
            if hasattr(psutil, 'Process'):
                current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                memory_usage = current_memory - memory_usage_start
                memory_info = f", 内存使用增加: {memory_usage:.1f}MB"
            
            print(f"已加载 {i+len(batch_files)}/{len(all_files)} 个文件, 当前数据大小: {len(precomputed_data)} 个节点")
            print(f"批次加载耗时: {batch_time:.2f}秒{memory_info}")
    
    if failed_files:
        print(f"警告: 加载 {len(failed_files)} 个文件失败: {failed_files[:5]}...")
    
    print(f"成功加载了 {len(precomputed_data)} 个节点的数据")
    return precomputed_data


# 修改主执行代码，实现自适应优化
if __name__ == "__main__":
    import time
    import torch.cuda
    import psutil
    
    print("使用极速优化版本进行图处理")
    
    # 系统资源检测
    num_cpus = os.cpu_count()
    
    # 获取内存信息，使用安全的方式
    memory_gb = 8.0  # 默认值
    try:
        if CPU_MEMORY > 0:
            memory_gb = CPU_MEMORY
        elif hasattr(psutil, 'virtual_memory') and psutil is not None:
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        elif sys.platform == 'darwin':  # macOS
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                memory_gb = int(result.stdout.strip()) / (1024 ** 3)
        elif sys.platform == 'linux':  # Linux
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        memory_gb = int(line.split()[1]) / (1024 ** 2)  # KB to GB
                        break
    except Exception as e:
        print(f"警告: 内存检测失败: {str(e)}, 使用默认值")
    
    # GPU检测
    gpu_available = torch.cuda.is_available()
    gpu_info = ""
    if gpu_available:
        gpu_info = f", GPU: {torch.cuda.get_device_name(0)}"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_info += f" ({gpu_memory:.1f}GB)"
    
    print(f"系统资源: {num_cpus} CPU核心, {memory_gb:.1f}GB 内存{gpu_info}")
    
    # 加载测试节点
    center_nodes = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))
    print(f"加载了 {len(center_nodes)} 个中心节点")
    
    # 自动检测并配置最佳参数
    # 设置最大子图大小 - 根据内存自动调整
    if memory_gb < 16:
        max_subgraph_size = 300  # 低内存设备
    elif memory_gb < 32:
        max_subgraph_size = 500  # 中等内存设备
    else:
        max_subgraph_size = 800  # 高内存设备
    
    # 设置工作进程数 - 保留核心给系统
    if num_cpus <= 4:
        num_workers = max(1, num_cpus - 1)  # 至少保留1个核心给系统
    else:
        num_workers = max(4, num_cpus // 2)  # 使用一半的CPU核心，至少4个
    
    # 基础批次大小 - 将在运行时自适应调整
    batch_size = 50
    save_dir = './precomputed/{}'.format(data_name)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否有断点续传
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        print(f"检测到检查点文件，将从上次中断点继续")
        try:
            import json
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                print(f"上次进度: 已处理 {len(checkpoint.get('processed_nodes', []))} 个节点")
                last_timestamp = checkpoint.get('timestamp', 0)
                time_diff = time.time() - last_timestamp
                print(f"距离上次运行: {time_diff/3600:.1f}小时前")
        except Exception as e:
            print(f"读取检查点失败: {str(e)}")
    
    # 估计最佳批次大小
    try:
        # 收集节点度数样本
        sample_size = min(len(center_nodes), 1000)
        sample_nodes = center_nodes[:sample_size]
        degrees = []
        for node in sample_nodes:
            if node < data.num_nodes:
                degree = torch_geometric.utils.degree(data.edge_index[0], num_nodes=data.num_nodes)[node].item()
                degrees.append(degree)
        
        # 计算自适应批次大小
        if degrees:
            avg_degree = sum(degrees) / len(degrees)
            # 估算批次大小 - 考虑度数分布
            if avg_degree < 50:
                adaptive_batch_size = 100  # 小度数节点用更大批次
            elif avg_degree < 200:
                adaptive_batch_size = 50   # 中等度数节点
            else:
                adaptive_batch_size = 30   # 高度数节点用小批次
                
            batch_size = adaptive_batch_size
            print(f"基于节点度数分布自适应设置批次大小: {batch_size} (平均度数: {avg_degree:.1f})")
    except Exception as e:
        print(f"自适应批次大小计算失败: {str(e)}")
    
    print(f"开始预计算，参数配置:")
    print(f"- 最大子图大小: {max_subgraph_size}")
    print(f"- 工作进程数: {num_workers}")
    print(f"- 批次大小: {batch_size}")
    print(f"- 保存目录: {save_dir}")
    print(f"- GPU加速: {'启用' if gpu_available else '禁用'}")

    # 确认并开始处理
    print("数据预处理将开始，这可能需要一段时间...")
    
    start_time = time.time()
    precompute_in_batches(data, center_nodes, L, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        save_dir=save_dir,
                        max_subgraph_size=max_subgraph_size)
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"预计算完成，总耗时: {hours}小时{minutes}分{seconds}秒")

    # 验证加载过程 - 使用惰性加载模式以减少内存使用
    print("验证数据加载...")
    lazy_loader = load_precomputed(save_dir, batch_size=5, lazy_loading=True)
    print(f"验证完成: 索引了 {len(lazy_loader)} 个节点")
    
    # 测试几个节点以确认加载功能正常
    test_count = min(5, len(center_nodes))
    if test_count > 0:
        print(f"测试加载 {test_count} 个随机节点...")
        success_count = 0
        for i in range(test_count):
            test_node = center_nodes[i]
            if test_node in lazy_loader:
                try:
                    node_data = lazy_loader[test_node]
                    print(f"节点 {test_node} 加载成功，子图大小: {node_data['subg_size']}")
                    success_count += 1
                except Exception as e:
                    print(f"节点 {test_node} 加载失败: {str(e)}")
            else:
                print(f"节点 {test_node} 不在预计算数据中")
        
        print(f"测试结果: {success_count}/{test_count} 节点加载成功")
                
        # 报告缓存统计
        lazy_loader.report_stats()
        
    print("预计算处理完成！")
    
    # 清理内存
    del data, lazy_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
