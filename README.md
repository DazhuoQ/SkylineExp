# SkylineExp

This is the repo for the paper "Generating Skyline Explanations for Graph Neural Networks".

## 1. Prepare the environment

First, create a virtual environment for the project:
`conda create -n skyexp python==3.10`

Then install the needed packages: 
`pip install -r requirements.txt`

## 2. Skyline Explanation Experiments

Train the GNN model `./train.sh config.yaml train_results/`

Run this script: `./run.sh config.yaml results/`

## 3. Parallel Experiments

First, preprocess the dataset: `python -m src.find_test_nodes && python -m src.pre_edges && python -m src.partalg`.

Use
`python -m src.paraalg`
to execute the parallel algorithm.

The outputs look like this: <img width="1134" alt="Screenshot 2025-04-25 at 15 16 28" src="https://github.com/user-attachments/assets/67f5e185-b81f-4b8b-9067-a606d51e26d1" />


Modify `m` in `config.yaml` to run different settings. 
