import numpy as np
import math
from collections import deque, defaultdict
import networkx as nx
import copy
from tqdm.std import tqdm
import random

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from src.plot import plot_L_hop_subg


class ApxSXI:
    
    def __init__(self, G, model, VT, k, L, epsilon):
        self.G = G # original graph
        self.model = model  # gnn model
        self.k = k  # size of the skyline set
        self.VT = VT  # test nodes
        self.L = L - 1 # num of gnn layers
        self.epsilon = epsilon
        self.k_sky_lst = []

        self.ipf_lst = []
        self.igd_lst = []
        self.ms_lst = []
        self.ipf = 0
        self.igd = np.inf


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


    def generate_k_skylines(self):
        edge_index = self.G.edge_index
        for vt in tqdm(self.VT, desc='num VT'):
            vt = vt.item()
            _, _, _, original_edge_mask = k_hop_subgraph(vt, self.L, edge_index, relabel_nodes=False)
            selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
            subg_size = selected_edge_positions.size(0)

            _, _, _, l2_edge_mask = k_hop_subgraph(vt, 1, edge_index, relabel_nodes=False)

            DRG = defaultdict(list)
            k_sky = []
            cur_v = vt
            epoch = 4
            visited = []
            s_0 = torch.zeros(edge_index.size(1), dtype=torch.bool)

            while epoch > 1:
                edge_size = s_0.sum().item() + 1

                source_edges = (edge_index[0] == cur_v)
                target_edges = (edge_index[1] == cur_v)
                involving_edges = source_edges | target_edges
                edge_positions = involving_edges.nonzero(as_tuple=True)[0]


                for edge_pos in edge_positions:
                    if edge_pos in visited:
                        continue
                    s = s_0.clone()
                    s[edge_pos] = True
                    fplus, fminus, factual, counterfactual = self.compute_fidelity(vt, s, original_edge_mask)
                    if not (factual or counterfactual):
                        continue
                    p_s = [fplus, fminus, 1-(math.log(edge_size)/math.log(subg_size))]
                    idx_s = []
                    for i in range(len(p_s)-1):
                        if p_s[i] <= 0:
                            tmp = math.floor(math.log(1e-6,(1+self.epsilon)))
                        else:
                            tmp = math.floor(math.log(p_s[i],(1+self.epsilon)))
                        idx_s.append(tmp)
                    idx_s = "".join(str(i) for i in idx_s)
        
                    DRG = self.update_sx(idx_s, DRG, s)
                
                epoch = epoch - 1
                e_star = random.choice(edge_positions)
                
                visited.append(e_star)
                s_0[e_star] = True

                source_node = edge_index[0][e_star]
                target_node = edge_index[1][e_star]
                if source_node.item() == cur_v:
                    neighbor_u = target_node
                else:
                    neighbor_u = source_node
                cur_v = neighbor_u

            k_sky = [DRG[key][0] for key in list(DRG.keys())]
            if len(k_sky) == 0:
                k_sky.append(l2_edge_mask)
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


