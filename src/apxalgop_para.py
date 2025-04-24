import numpy as np
import math
from collections import deque, defaultdict
import networkx as nx
import copy
from tqdm.std import tqdm
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

# from src.plot import plot_L_hop_subg


class ParaApxSXOP:
    
    def __init__(self, G, model, VT, k, L, epsilon):
        self.G = G # original graph
        self.model = model  # gnn model
        self.k = k  # size of the skyline set
        self.VT = VT  # test nodes
        self.L = L  # num of gnn layers
        self.epsilon = epsilon
        self.k_sky_lst = []

        self.ipf_lst = []
        self.igd_lst = []
        self.ms_lst = []
        self.ipf = 0
        self.igd = np.inf


    def get_edge_sets_by_hop(self, vt):

        L = self.L
        edge_index = self.G.edge_index

        node_idx, edge_index_sub, _, original_edge_mask = k_hop_subgraph(vt, L, edge_index, relabel_nodes=False)
        ori_mask = original_edge_mask
        selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
        subg_size = selected_edge_positions.size(0)
        
        hop_distances = {node.item(): float('inf') for node in node_idx}
        hop_distances[vt] = 0
        queue = deque([vt])
        
        while queue:
            current_node = queue.popleft()
            current_hop = hop_distances[current_node]
            
            for edge_idx in selected_edge_positions:
                src, dst = edge_index[:, edge_idx]
                if src.item() == current_node:
                    if hop_distances[dst.item()] == float('inf'):
                        hop_distances[dst.item()] = current_hop + 1
                        queue.append(dst.item())
                elif dst.item() == current_node:
                    if hop_distances[src.item()] == float('inf'):
                        hop_distances[src.item()] = current_hop + 1
                        queue.append(src.item())
        
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


    def compute_fidelity(self, node_idx, edge_mask, ori_mask):

        model = self.model
        data = self.G
        original_edge_index = self.G.edge_index

        with torch.no_grad():
            y_original = F.softmax(model(data.x, original_edge_index[:, ori_mask]), dim=1)[node_idx]
            original_label = y_original.argmax()
            y_original = y_original[original_label]

        mask_edge_index = original_edge_index[:, edge_mask]

        with torch.no_grad():
            y_subgraph = F.softmax(model(data.x, mask_edge_index), dim=1)[node_idx]
            subgraph_label = y_subgraph.argmax()
            y_subgraph = y_subgraph[original_label]

        complementary_edge_index = original_edge_index[:, ~edge_mask]
        
        with torch.no_grad():
            y_complementary = F.softmax(model(data.x, complementary_edge_index), dim=1)[node_idx]
            complementary_label = y_complementary.argmax()
            y_complementary = y_complementary[original_label]

        factual = (subgraph_label == original_label)

        counterfactual = (complementary_label != original_label)

        fidelity_plus = y_original - y_complementary
        fidelity_plus = fidelity_plus.item()
        fidelity_minus = y_subgraph - y_original + 1
        fidelity_minus = fidelity_minus.item()

        return fidelity_plus, fidelity_minus, factual, counterfactual


    def find_argmin_s(self, DRG):
        min_key = None
        min_size = float('inf')
        for key, value in DRG.items():
            current_size = len(value)
            if current_size < min_size:
                min_size = current_size
                min_key = key
        return min_key, min_size


    def update_sx(self, idx_s, DRG, s):
        idx_s = str(idx_s)
        if idx_s not in DRG:
            if len(DRG) < self.k:
                DRG[idx_s] = [s]
            else:
                s_overline, min_size = self.find_argmin_s(DRG)
                tot = sum(len(value) for value in DRG.values())
                if min_size < tot/self.k:
                    del DRG[s_overline]
                    DRG[idx_s].append(s)
        elif DRG[idx_s][0].sum().item() > s.sum().item():
            DRG[idx_s].insert(0,s)
        elif DRG[idx_s][0].sum().item() <= s.sum().item():
            DRG[idx_s].append(s)
        return DRG


    def generate_k_skylines(self, vt):
        DRG = defaultdict(list)
        k_sky = []
        vt = vt.item()
        edges_by_hop, edge_masks_by_hop, subg_size, ori_mask = self.get_edge_sets_by_hop(vt)
        for hop in range(self.L, 0, -1):
            s_0 = edge_masks_by_hop[hop].clone()
            E_l = edges_by_hop[hop]
            while len(E_l) != 0:
                t_star = (None, [np.inf, np.inf, np.inf])

                if len(E_l) > 10:
                    E_l = random.sample(E_l, 8)

                edge_size = s_0.sum().item() - 1
                if edge_size == 0:
                    break
                fplus_0, fminus_0, factual_0, counterfactual_0 = self.compute_fidelity(vt, s_0, ori_mask)
                for edge_pos in E_l: # iter E_l
                    s = s_0.clone()
                    s[edge_pos] = False
                    fplus, fminus, factual, counterfactual = self.compute_fidelity(vt, s, ori_mask)
                    if not (factual or counterfactual):
                        continue
                    t = (edge_pos, [fplus_0-fplus, fminus_0-fminus, -1/subg_size])
                    p_s = [1-fplus, 1-fminus, (math.log(edge_size)/math.log(subg_size))]
                    idx_s = []
                    for i in range(len(p_s)-1):
                        if p_s[i] <= 0:
                            tmp = math.floor(math.log(1e-6,(1+self.epsilon)))
                        else:
                            tmp = math.floor(math.log(p_s[i],(1+self.epsilon)))
                        idx_s.append(tmp)
                    idx_s = "".join(str(i) for i in idx_s)
                    DRG = self.update_sx(idx_s, DRG, s)
                    if np.mean(t_star[1]) > np.mean(t[1]):
                        t_star = t
                if t_star[0] == None:
                    break
                
                E_l.remove(t_star[0])
                s_0[t_star[0]] = False
        k_sky = [DRG[key][0] for key in list(DRG.keys())]
        self.k_sky_lst.append((vt, k_sky))


    def IPF(self):
        for vt, k_sky in self.k_sky_lst:
            _, _, _, original_edge_mask = k_hop_subgraph(vt, self.L, self.G.edge_index, relabel_nodes=False)
            selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
            subg_size = selected_edge_positions.size(0)
            score_lst = []
            for mask in k_sky:
                fidelity_plus, fidelity_minus, _, _ = self.compute_fidelity(vt, mask, original_edge_mask)
                conc = 1 - (math.log(mask.sum().item()) / math.log(subg_size))
                tmp = (fidelity_plus + fidelity_minus + conc)/3
                score_lst.append(tmp)
            score = np.mean(score_lst)
            self.ipf_lst.append((vt, score))
            self.ipf = np.mean([score for vt, score in self.ipf_lst])




