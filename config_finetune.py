import argparse
import json
from typing import Dict, Any

from utils import str2bool


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="配置解析器")

    # 基本参数
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts/", help="检查点保存目录")
    parser.add_argument("--ckpt_path", type=str, default="checkpoint.pt", help="检查点文件名")
    parser.add_argument("--sample", type=str2bool, default=False, help="是否采样数据集")
    parser.add_argument("--trainable", type=str2bool, default=True, help="模型是否可训练")
    parser.add_argument("--lora", type=str2bool, default=False, help="是否使用LoRA")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="混合精度训练模式")
    parser.add_argument("--dataset", type=str, default="movielens", choices=["movielens", "bookcrossing", "amazon"], help="数据集名称")
    parser.add_argument("--backbone", type=str, default="DCNv2", help="骨干网络架构")
    parser.add_argument("--llm", type=str, default="tinybert", choices=["distilbert", "bert", "roberta", "roberta-large", "tinybert"], help="语言模型")
    parser.add_argument("--describe", type=str, default="temp", help="实验描述")
    parser.add_argument("--optimizer", type=str, default="Adam", help="优化器")

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--embedding_dim", type=int, default=32, help="嵌入维度")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--delta", type=float, default=0, help="最小改进阈值")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout率")
    parser.add_argument("--lr1", type=float, default=1e-3, help="优化器1学习率")
    parser.add_argument("--lr2", type=float, default=1e-4, help="优化器2学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--t", type=float, default=0.3, help="温度参数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")

    args = parser.parse_args()

    # 设置路径
    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    # 数据集配置
    dataset_configs: Dict[str, Dict[str, Any]] = {
        'movielens': {
            'data_path': f"{args.load_prefix_path}data/ml-1m/",
            'max_length': 30,
            'text_path': f"{args.load_prefix_path}data/ml-1m/text_data.csv",
            'struct_path': f"{args.load_prefix_path}data/ml-1m/struct_data.csv"
        },
        'bookcrossing': {
            'data_path': f"{args.load_prefix_path}data/bookcrossing/",
            'max_length': 100,
            'text_path': f"{args.load_prefix_path}data/bookcrossing/text_data_shuffle_lower.csv",
            'struct_path': f"{args.load_prefix_path}data/bookcrossing/struct_data2_shuffle.csv"
        },
        'amazon': {
            'data_path': f"{args.load_prefix_path}data/amazon/",
            'text_path': f"{args.load_prefix_path}data/amazon/text_data.csv",
            'struct_path': f"{args.load_prefix_path}data/amazon/struct2.csv",
            'max_length': 100
        }
    }

    # 设置数据集特定配置
    config = dataset_configs.get(args.dataset, {})
    for key, value in config.items():
        setattr(args, key, value)

    # 设置通用路径
    args.meta_path = f"{args.data_path}meta.json"

    # 语言模型配置
    llm_configs: Dict[str, Dict[str, Any]] = {
        'distilbert': {'model': "distilbert-base-uncased", 'dim': 768},
        'tinybert': {'model': "huawei-noah/TinyBERT_General_4L_312D", 'dim': 384},
        'opt': {'model': "facebook/opt-1.3b", 'dim': 2048},
        'bert': {'model': "bert-base-uncased", 'dim': 768},
        'roberta': {'model': "roberta-base", 'dim': 768},
        'roberta-large': {'model': "roberta-large", 'dim': 1024},
        'SFR': {'model': "SFR-Embedding-Mistral", 'dim': 4096}
    }

    # 设置语言模型配置
    llm_config = llm_configs.get(args.llm, {})
    args.text_encoder_model = f"{args.load_prefix_path}pretrained_models/{llm_config['model']}/"
    args.text_tokenizer = args.text_encoder_model
    args.text_embedding_dim = llm_config['dim']

    return args