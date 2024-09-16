import sys
import subprocess
import argparse
import os
from itertools import product

# Environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,0"

# Argument parsing
parser = argparse.ArgumentParser(description="Training script with various hyperparameters")
parser.add_argument('--dataset', type=str, default='movielens', choices=['movielens', 'bookcrossing', 'amazon'])
parser.add_argument('--backbone', type=str, default='AutoInt',
                    choices=['DeepFM', 'AutoInt', 'DCNv2', 'DCN', 'xDeepFM', 'PNN', 'widedeep'])
parser.add_argument('--llm', type=str, default='distilbert', choices=['bert', 'distilbert', 'roberta', 'roberta-large'])
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

# Constants
TARGET_PY_FILE = 'bert_embedding_finetune_ddp.py'
PREFIX = f'/home/yutao/.conda/envs/FLIP/bin/python -m torch.distributed.launch --nproc_per_node 5 --use_env --master_port 41012 {TARGET_PY_FILE}'.split()

# Hyperparameters
hyperparameters = {
    'optimizer': ['Adam'],
    # 'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # 'temperature': [0.3],
    'lr1': [5e-3],
    'lr2': [1e-4],
    'batch_size': [256],
    'epochs': [args.epochs],
    'dataset': ['movielens'],
    'backbone': ['DeepFM', 'AutoInt', 'DCNv2','DCN', 'xDeepFM', 'PNN', 'widedeep'],#
    'trainable': ['True'],
    'lora': ['False']
}


def run_training(params):
    cmd = PREFIX + [f'--{k}={v}' for k, v in params.items()]
    subprocess.run(cmd)


# Generate all combinations of hyperparameters
param_combinations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

# Run training for each combination
for params in param_combinations:
    run_training(params)
