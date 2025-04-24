import random
import os
import numpy as np
import yaml
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, FacebookPagePage, AmazonProducts, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def dataset_func(config):
    
    data_dir = "./datasets"
    data_name = config['data_name']
    data_size = config['data_size']
    num_class = config['output_dim']
    num_test = config['num_test']
    random_seed = config['random_seed']
    os.makedirs(data_dir, exist_ok=True)
    set_seed(random_seed)

    if data_name == "FacebookPage":
        dataset = FacebookPagePage(root="./datasets/FacebookPagePage")
        data = dataset[0]
        num_nodes = data.x.size(0)

        # Create new masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Example: 60% train, 20% val, 20% test
        num_train = int(0.6 * num_nodes)
        num_val = num_nodes - num_train - num_test

        # Set the masks
        train_mask[:num_train] = 1
        val_mask[num_train:num_train + num_val] = 1
        test_mask[num_train + num_val:] = 1

        # Assign the new masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(data)
        return data

    if data_name == "AmazonComputers":
        
        # Load the Amazon Computers dataset
        dataset = Amazon(root='./datasets/', name='computers')
        data = dataset[0]
        num_nodes = data.x.size(0)

        # Create new masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Example: 60% train, 20% val, 20% test
        num_train = int(0.6 * num_nodes)
        num_val = num_nodes - num_train - num_test

        # Set the masks
        train_mask[:num_train] = 1
        val_mask[num_train:num_train + num_val] = 1
        test_mask[num_train + num_val:] = 1

        # Assign the new masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(data)
        return data
    
    if data_name == "arxiv":
        dataset = PygNodePropPredDataset(root='./datasets/', name='ogbn-arxiv')
        data = dataset[0]
        num_nodes = data.x.size(0)
        data.y = data.y[:,0]

        # Create new masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Example: 60% train, 20% val, 20% test
        num_train = int(0.6 * num_nodes)
        num_val = num_nodes - num_train - num_test

        # Set the masks
        train_mask[:num_train] = 1
        val_mask[num_train:num_train + num_val] = 1
        test_mask[num_train + num_val:] = 1

        # Assign the new masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(data)
        return data

    if data_name == "syn":
        dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=300, num_edges=5),
            motif_generator='house',
            num_motifs=80,
            transform=T.Constant(),
        )
        data = dataset[0]
        num_nodes = data.x.size(0)

        # Create new masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Example: 60% train, 20% val, 20% test
        num_train = int(0.6 * num_nodes)
        num_val = num_nodes - num_train - num_test

        # Set the masks
        train_mask[:num_train] = 1
        val_mask[num_train:num_train + num_val] = 1
        test_mask[num_train + num_val:] = 1

        # Assign the new masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(data)
        return data
    

    num_train_per_class = (data_size - num_test)//num_class
    data = Planetoid(root=data_dir, name=data_name, split='random', num_train_per_class=num_train_per_class, num_val=0, num_test=num_test)[0]
    return data


def get_save_path(dataset, apx_name):
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Define base directory for results relative to the script's directory
    base_results_directory = os.path.join(current_directory, "results")
    os.makedirs(base_results_directory, exist_ok=True)

    dataset_path = os.path.join(base_results_directory, dataset)
    os.makedirs(dataset_path, exist_ok=True)

    method_path = os.path.join(dataset_path, apx_name)
    os.makedirs(method_path, exist_ok=True)

    return method_path

