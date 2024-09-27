import sys
import subprocess
import argparse
import os
from itertools import product

# Environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,0"

# Argument parsing
parser = argparse.ArgumentParser(description="Training script with various hyperparameters")
parser.add_argument('--dataset', type=str, default='bookcrossing', choices=['movielens', 'bookcrossing', 'amazon'])
parser.add_argument('--backbone', type=str, default='AutoInt',
                    choices=['DeepFM', 'AutoInt', 'DCNv2', 'DCN', 'xDeepFM', 'PNN', 'widedeep'])
parser.add_argument('--llm', type=str, default='roberta')
parser.add_argument('--epochs', type=int, default=1)
args = parser.parse_args()

# Constants
TARGET_PY_FILE = 'bert_embedding_finetune_ddp.py'
PREFIX = f'/home/yutao/.conda/envs/FLIP/bin/python -m torch.distributed.launch --nproc_per_node 5 --use_env --master_port 41012 {TARGET_PY_FILE}'.split()

# Hyperparameters
hyperparameters = {
    'optimizer': ['Adam'],
    'embedding_dim':[16,64],
    'gamma':[1],
    'llm': ['distilbert'],
    'lr1': [1e-3],
    'lr2': [1e-4],
    't1': [0.3],
    't3': [0.3],
    'batch_size': [256],
    'epochs': [args.epochs],
    'dataset': ['bookcrossing'],  # 'bookcrossing',
    'backbone': ['DCNv2'],  # ,, 'AutoInt','xDeepFM', 'PNN', 'widedeep','DeepFM','DCNv2',
    'trainable': ['True'],
    'describe': ["超参数敏感性分析"]
}


def run_training(params):
    cmd = PREFIX + [f'--{k}={v}' for k, v in params.items()]
    subprocess.run(cmd)


# Generate all combinations of hyperparameters
param_combinations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

# Filter out combinations where alpha + beta >= 1
# param_combinations = [params for params in param_combinations if float(params['alpha']) + float(params['beta']) < 1]

# Run training for each combination
for params in param_combinations:
    print("总共有{}个参数组合".format(len(param_combinations)))
    print("当前参数组合:", params)
    print("当前为第{}个参数组合".format(param_combinations.index(params) + 1))
    run_training(params)
