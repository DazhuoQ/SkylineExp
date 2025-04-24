from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import metis
import networkx as nx
from collections import defaultdict

# Example dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Convert to NetworkX (METIS needs this)
nxG = to_networkx(data, to_undirected=True)

# Partition the full graph into 5 parts
_, parts = metis.part_graph(nxG, nparts=5)

# Suppose this is your test node set
test_nodes = data.test_mask.nonzero(as_tuple=True)[0].tolist()

# Group test nodes by their partition assignment
clusters = defaultdict(list)
for node in test_nodes:
    clusters[parts[node]].append(node)

print("Clusters of test nodes:", dict(clusters))
