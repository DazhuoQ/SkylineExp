import os
import sys
import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import is_undirected
from torch_geometric.utils import k_hop_subgraph

from src.utils import *
from src.model import get_model
from src.apxalgop import ApxSXOP
# from src.apxalgi import ApxSXI
# from src.divalg import DivSX
# from src.linearalg import LinearALG
# from src.individualalg import IndividualALG


def main(config_file, output_dir):

    # Load configuration
    config = load_config(config_file)
    data_name = config['data_name']
    model_name = config['model_name']
    random_seed = config['random_seed']
    L = config['L']
    k = config['k']
    epsilon = config['epsilon']
    exp_name = config['exp_name']
    VT = config['VT']
    # vt = config['vt']
    
    # Save experiment settings
    print('Seed: '+str(config['random_seed']))
    print('Exp: '+str(config['exp_name']))
    print('Dataset: '+str(config['data_name']))
    print('k: '+str(config['k']))
    print('Model: '+str(config['model_name']))
    print('L: '+str(config['L']))

    set_seed(random_seed)

    # Get input graph
    data = dataset_func(config)

    # Get the VT
    VT = torch.tensor(VT)

    # Ready the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config)
    model.load_state_dict(torch.load('models/{}_{}_model.pth'.format(data_name, model_name), weights_only=False, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    # experiments
    if exp_name == 'ksx':
        VT = torch.load('./datasets/{}/test_nodes.pt'.format(data_name))
        save_dir = './precomputed/{}'.format(data_name)
        precomputed_data = load_precomputed(save_dir)
        algorithm = ApxSXOP(G = data, 
                          model = model, 
                          data_name = data_name,
                          VT = VT, 
                          k = k, 
                          L = L,
                          epsilon = epsilon,
                          precomputed_data = precomputed_data
                          )
        start_time = time.time()
        algorithm.generate_k_skylines()
        end_time = time.time()

        path = get_save_path(data_name, exp_name)
        filename = f'{str(k)}_{model_name}_{str(L)}.pt'
        torch.save(algorithm.k_sky_lst, os.path.join(path, filename))

        print("Running Time: {:.2f} seconds".format(end_time - start_time))
        algorithm.IPF()
        print('IPF scores: {:.2f}'.format(algorithm.ipf))

        print('ok')

    if exp_name == 'insert':
        algorithm = ApxSXI(G = data, 
                          model = model, 
                          VT = VT, 
                          k = k, 
                          L = L,
                          epsilon = epsilon)
        start_time = time.time()
        algorithm.generate_k_skylines()
        end_time = time.time()

        path = get_save_path(data_name, exp_name)
        filename = f'{str(k)}_{model_name}_{str(L)}.pt'
        torch.save(algorithm.k_sky_lst, os.path.join(path, filename))
        
        print("Running Time: {:.2f} seconds".format(end_time - start_time))
        algorithm.IPF()
        print('IPF scores: {:.2f}'.format(algorithm.ipf))

        print('ok')

    if exp_name == 'div':
        algorithm = DivSX(G = data, 
                          model = model, 
                          VT = VT, 
                          k = k, 
                          L = L,
                          epsilon = epsilon,
                          beta = config['beta'],
                          alpha = config['alpha'],
                          )
        start_time = time.time()
        algorithm.generate_k_skylines()
        end_time = time.time()

        path = get_save_path(data_name, exp_name)
        filename = f'{str(k)}_{model_name}_{str(L)}.pt'
        torch.save(algorithm.k_sky_lst, os.path.join(path, filename))

        print("Running Time: {:.2f} seconds".format(end_time - start_time))
        algorithm.IPF()
        print('IPF scores: {:.2f}'.format(algorithm.ipf))

        print('ok')

    if exp_name == 'linear':
        algorithm = LinearALG(G = data, 
                          model = model, 
                          VT = VT, 
                          k = k, 
                          L = L,
                          epsilon = epsilon,
                          beta = config['beta'],
                          alpha = config['alpha'],
                          )
        start_time = time.time()
        algorithm.generate_k_skylines()
        end_time = time.time()

        path = get_save_path(data_name, exp_name)
        filename = f'{str(k)}_{model_name}_{str(L)}.pt'
        torch.save(algorithm.k_sky_lst, os.path.join(path, filename))

        print("Running Time: {:.2f} seconds".format(end_time - start_time))
        # algorithm.IPF()
        # print('IPF scores: {:.2f}'.format(algorithm.ipf))

        print('ok')

    if exp_name == 'individual':
        algorithm = IndividualALG(G = data, 
                          model = model, 
                          VT = VT, 
                          k = k, 
                          L = L,
                          epsilon = epsilon)
        start_time = time.time()
        algorithm.generate_k_skylines()
        end_time = time.time()

        path = get_save_path(data_name, exp_name)
        filename = f'{str(k)}_{model_name}_{str(L)}.pt'
        torch.save(algorithm.k_sky_lst, os.path.join(path, filename))

        print("Running Time: {:.2f} seconds".format(end_time - start_time))
        # algorithm.IPF()
        # print('IPF scores: {:.2f}'.format(algorithm.ipf))

        print('ok')
    
    if exp_name == '3d':
        algorithm = ApxSXI(G = data, 
                          model = model, 
                          VT = torch.tensor([vt]), 
                          k = k, 
                          L = L,
                          epsilon = epsilon)
        start_time = time.time()
        algorithm.generate_k_skylines()
        end_time = time.time()

        # path = get_save_path(data_name, exp_name)
        filename = f'{str(vt)}_results.pt'
        torch.save(algorithm.k_sky_lst, filename)

        print('ok')        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python experiment.py <config_file> <output_dir>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(config_file, output_dir)

