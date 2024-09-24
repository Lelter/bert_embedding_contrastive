import argparse
from utils import str2bool
import wandb
import os


# os.chdir('/data/yyt/bert_embedding_contrastive/bert_embedding_contrastive')
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ckpt_dir", type=str, default="ckpts/")
    parser.add_argument("--ckpt_path", type=str, default="checkpoint.pt")

    parser.add_argument("--sample", type=str2bool, default=False, help="sample dataset")
    parser.add_argument("--trainable", type=str2bool, default=True, help="sample dataset")

    # mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16")

    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument('--backbone', default='DCNv2', type=str, help='')
    parser.add_argument("--llm", type=str, default='distilbert', help='language model')
    parser.add_argument("--describe", type=str, default='设备3')
    parser.add_argument("--optimizer", type=str, default='Adam')
    # temperature
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--delta", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=1e-5)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--t1", type=float, default=0.3)
    parser.add_argument("--t3", type=float, default=0.3)
    parser.add_argument("--step_size", type=float, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    if args.dataset == 'movielens':
        args.data_path = args.load_prefix_path + "data/ml-1m/"
        args.max_length = 30
        args.text_path = args.data_path + "text_data.csv"
        args.struct_path = args.data_path + "struct_data.csv"

    elif args.dataset == 'bookcrossing':
        args.data_path = args.load_prefix_path + "data/bookcrossing/"
        args.max_length = 100
        args.text_path = args.data_path + "text_data_shuffle_lower.csv"
        args.struct_path = args.data_path + "struct_data2_shuffle.csv"

    elif args.dataset == 'amazon':
        args.data_path = args.load_prefix_path + "data/amazon/"
        args.max_length = 100
        args.text_path = args.data_path + "text_data.csv"
        args.struct_path = args.data_path + "struct2.csv"
    args.feat_count_path = args.data_path + 'feat_count.pt'
    args.meta_path = args.data_path + 'meta.json'
    if args.llm == 'smallbert':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/prajjwal1/bert-small/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/prajjwal1/bert-small/"
        args.text_embedding_dim = 512
    if args.llm == 'mediumbert':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/prajjwal1/bert-medium/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/prajjwal1/bert-medium/"
        args.text_embedding_dim = 512
    if args.llm == 'largebert':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/google-bert/bert-large-uncased/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/google-bert/bert-large-uncased/"
        args.text_embedding_dim = 1024
    if args.llm == 'distilbert':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/distilbert-base-uncased/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/distilbert-base-uncased/"
        args.text_embedding_dim = 768
    if args.llm == 'tinybert':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/huawei-noah/TinyBERT_General_4L_312D/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/huawei-noah/TinyBERT_General_4L_312D/"
        args.text_embedding_dim = 384
    elif args.llm == 'bert':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/bert-base-uncased/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/bert-base-uncased/"
        args.text_embedding_dim = 768
    elif args.llm == 'roberta':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/roberta-base/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/roberta-base/"
        args.text_embedding_dim = 768
    elif args.llm == 'roberta-large':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/roberta-large/"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/roberta-large/"
        args.text_embedding_dim = 1024
    elif args.llm == 'opt':
        args.text_encoder_model = args.load_prefix_path + "pretrained_models/facebook/opt-1.3b"
        args.text_tokenizer = args.load_prefix_path + "pretrained_models/facebook/opt-1.3b"
        args.text_embedding_dim = 2048

    args.sample_ration = 0.01
    for k, v in vars(args).items():
        print(k, '=', v)

    return args
