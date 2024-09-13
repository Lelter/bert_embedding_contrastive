import argparse
from utils import str2bool
import json


def create_parser():
    parser = argparse.ArgumentParser(description="Configuration parser for ML project")

    # Add arguments with more descriptive help messages and appropriate types
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    parser.add_argument("--ckpt_dir", type=str, default="ckpts/", help="Directory to save checkpoints")
    parser.add_argument("--ckpt_path", type=str, default="checkpoint.pt", help="Checkpoint file name")
    parser.add_argument("--sample", type=str2bool, default=False, help="Whether to sample the dataset")
    parser.add_argument("--trainable", type=str2bool, default=True, help="Whether the model is trainable")
    parser.add_argument("--lora", type=str2bool, default=False, help="Whether use LoRA")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision training mode")
    parser.add_argument("--dataset", type=str, default="amazon", choices=["movielens", "bookcrossing", "amazon"],
                        help="Dataset name")
    parser.add_argument("--backbone", type=str, default="DCNv2", help="Backbone model architecture")
    parser.add_argument("--llm", type=str, default="bert",
                        choices=["distilbert", "bert", "roberta", "roberta-large", "SFR"], help="Language model to use")
    parser.add_argument("--describe", type=str, default="设备3", help="Description of the experiment")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer for training")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of embeddings")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--delta", type=float, default=0, help="Minimum change to qualify as improvement")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Learning rate for optimizer 1")
    parser.add_argument("--lr2", type=float, default=1e-4, help="Learning rate for optimizer 2")
    parser.add_argument("--t", type=float, default=0.3, help="Temperature parameter")
    parser.add_argument("--step_size", type=int, default=3, help="Step size for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for regularization")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")

    args = parser.parse_args()

    # Set paths
    args.load_prefix_path = "../"
    args.output_prefix_path = '../'

    # Dataset-specific configurations
    dataset_configs = {
        'movielens': {
            'data_path': args.load_prefix_path + "data/ml-1m/",
            'max_length': 30,
            'text_path': args.load_prefix_path + "data/ml-1m/"+"text_data.csv",
            'struct_path': args.load_prefix_path + "data/ml-1m/"+"struct_data.csv"
        },
        'bookcrossing': {
            'data_path': args.load_prefix_path + "data/bookcrossing/",
            'max_length': 100,
            'text_path': args.load_prefix_path + "data/bookcrossing/"+"text_data_shuffle.csv",
            'struct_path': args.load_prefix_path + "data/bookcrossing/"+"struct_data2_shuffle.csv"
        },
        'amazon': {
            'data_path': args.load_prefix_path + "data/amazon/",
            'text_path': args.load_prefix_path + "data/amazon/" + "text_data.csv",
            'struct_path': args.load_prefix_path + "data/amazon/" + "struct.csv",
            'max_length': 100
        }
    }

    # Set dataset-specific configurations
    config = dataset_configs.get(args.dataset, {})
    for key, value in config.items():
        setattr(args, key, value)

    # Set common paths
    args.feat_count_path = args.data_path + 'feat_count.pt'
    args.meta_path = args.data_path + 'meta.json'

    # Language model configurations
    llm_configs = {
        'distilbert': {'model': "distilbert-base-uncased", 'dim': 768},
        'bert': {'model': "bert-base-uncased", 'dim': 768},
        'roberta': {'model': "roberta-base", 'dim': 768},
        'roberta-large': {'model': "roberta-large", 'dim': 1024},
        'SFR': {'model': "SFR-Embedding-Mistral", 'dim': 4096}
    }

    # Set language model configurations
    llm_config = llm_configs.get(args.llm, {})
    args.text_encoder_model = args.load_prefix_path + f"pretrained_models/{llm_config['model']}/"
    args.text_tokenizer = args.text_encoder_model
    args.text_embedding_dim = llm_config['dim']

    args.sample_ration = 0.01

    # Print all arguments
    print(json.dumps(vars(args), indent=2))

    return args