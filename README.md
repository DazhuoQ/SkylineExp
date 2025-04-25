# SkylineExp

This is the repo for the paper "Generating Skyline Explanations for Graph Neural Networks".

## 1. Prepare the environment

First, create a virtual environment for the project:
`conda create -n skyexp python==3.10`

Then install the needed packages: 
`pip install -r requirements.txt`

## 2. Parallel Experiments

Use
`python -m src.paraalg`
to execute the parallel algorithm.

The outputs look like this: <img width="1134" alt="Screenshot 2025-04-25 at 15 16 28" src="https://github.com/user-attachments/assets/67f5e185-b81f-4b8b-9067-a606d51e26d1" />


Modify `m` in `config.yaml` to run different settings. 
