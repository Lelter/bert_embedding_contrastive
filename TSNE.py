import os
import os
import warnings

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from config_pretrain_modal import create_parser
from datasets import BertCTRDataset
from models.LLM_CTR_modal_model import bertCTRModel
from preprocessing.inputs import SparseFeat

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, pipeline
# from model.MaskCTR_ddp import MaskCTR
from utils import EarlyStopping

from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
from accelerate.logging import get_logger
import logging


# sns.set_theme()
def process_struct_data(data_source, train, val, test, data):
    # 对结构化数据进行处理
    embedding_dim = cfg.embedding_dim
    sparse_features = {
        'movielens': ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'title', 'genres'],
        'bookcrossing': ['userId', 'ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'location', 'publisher',
                         'age'],
        'amazon': ['user_id', 'item_id', 'category', 'brand', 'title', ]
    }[data_source]
    sparse_features_text = {
        'movielens': ['gender', 'age', 'occupation', 'zip', 'title', 'genres'],
        'bookcrossing': ['bookTitle', 'bookAuthor', 'yearOfPublication', 'location', 'publisher', 'age'],
        'amazon': ['category', 'brand', 'title', ]
    }[data_source]

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                              for feat in sparse_features]
    sparse_text_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                   for feat in sparse_features_text]
    sparse_feature_columns = sparse_feature_columns
    sparse_all_feature_columns = sparse_feature_columns + sparse_text_feature_columns
    label_encoders = {feat: LabelEncoder().fit(data[feat]) for feat in sparse_features}
    for feat in sparse_features:
        train[feat] = label_encoders[feat].transform(train[feat])
        test[feat] = label_encoders[feat].transform(test[feat])
        val[feat] = label_encoders[feat].transform(val[feat])

    train_model_input = {name: train[name] for name in sparse_features}
    test_model_input = {name: test[name] for name in sparse_features}
    val_model_input = {name: val[name] for name in sparse_features}

    return sparse_feature_columns, sparse_all_feature_columns, train_model_input, val_model_input, test_model_input


def make_train_test_dfs(struct_data_path, text_data_path, seed):
    # 划分训练集和测试集
    length = 40000

    # 读取结构化数据和文本数据
    # struct_data = pd.read_csv(struct_data_path)
    struct_data = pd.read_csv(struct_data_path)[:length]
    # text_data = pd.read_csv(text_data_path)
    text_data = pd.read_csv(text_data_path)[:length]
    text_data['label'] = struct_data['label']

    def split_data(data, train_ratio=0.8, val_ratio=0.1):
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        return data.iloc[:train_size].copy(), data.iloc[train_size:train_size + val_size].copy(), data.iloc[
                                                                                                  train_size + val_size:].copy()

    train_struct, val_struct, test_struct = split_data(struct_data)
    train_text, val_text, test_text = split_data(text_data)

    return train_struct, val_struct, test_struct, train_text, val_text, test_text, struct_data


def build_loaders(struct_input, text_input, sparse_feature_columns, sparse_all_feature_columns, tokenizer,
                  shuffle=True):
    dataset = BertCTRDataset(
        struct_input,
        text_input,
        text_input["label"].values,
        # text_input['pseudo_label'].values,
        sparse_feature_columns=sparse_feature_columns,
        sparse_all_feature_columns=sparse_all_feature_columns,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
    return dataloader


def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            # logger.info(f'{name}: mean={param.grad.mean()}, std={param.grad.std()}')
            continue
        else:
            logger.info(f'{name}: None')


def main(cfg):
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    early_stopping = EarlyStopping(patience=cfg.patience, delta=cfg.delta, mode='max')
    data_type = cfg.dataset
    # ======================================================================
    train_struct, val_struct, test_struct, train_text, val_text, test_text, struct_data = make_train_test_dfs(
        cfg.struct_path,
        cfg.text_path, cfg.seed,
    )
    sparse_feature_columns, sparse_all_feature_columns, train_struct_input, val_struct_input, test_struct_input = \
        process_struct_data(data_type, train_struct, val_struct, test_struct, struct_data)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    train_loader = build_loaders(train_struct_input, train_text,
                                 sparse_feature_columns, sparse_all_feature_columns, tokenizer, True)
    test_loader = build_loaders(test_struct_input, test_text,
                                sparse_feature_columns, sparse_all_feature_columns, tokenizer, True)
    with open(cfg.meta_path) as fh:
        meta_data = json.load(fh)
    total_feature_num = meta_data['feature_num']
    model = bertCTRModel(cfg, text_embedding_dim=cfg.text_embedding_dim,
                         text_encoder_model=cfg.text_encoder_model,
                         sparse_feature_columns=sparse_feature_columns,
                         sparse_all_feature_columns=sparse_all_feature_columns,
                         struct_feature_num=total_feature_num,
                         )
    # optimizer = optim.Adagrad(model.parameters(), lr=cfg.lr,eps=1e-4)
    ablation_name = 'no'
    ablation_map = {
        'just_cross_loss': 'just_cross',
        'just_in_cross_loss': 'just_in_cross',
        'just_in_loss': 'just_in',
        'just_modal': 'just_modal_TSNE',
        'final': 'final',
        'no': 'no'
    }

    # 获取模型名称，若不存在则返回默认值'all'
    model_name = ablation_map.get(ablation_name, 'all')
    if model_name != 'no':
        model_path = f'ckpts/ablation/{cfg.dataset}/{cfg.llm}/{model_name}.pth'

        # 加载模型参数，关闭严格模式以防止某些键不匹配的错误
        model.load_state_dict(torch.load(model_path), strict=False)
    no_grad_params = {'text_encoder.model.pooler.dense.weight',
                      'text_encoder.model.pooler.dense.bias'}  # 冻结 accelerate需要有梯度
    for name, param in model.named_parameters():
        if name in no_grad_params:
            param.requires_grad = False
    model, train_loader = accelerator.prepare(model, train_loader)

    # TSNE可视化
    def extract_features(model, dataloader, device):
        model.eval()  # 模型进入评估模式
        text_embeddings = []
        rec_embeddings = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels.append(batch['label'].cpu().numpy())
                # 从模型中提取嵌入或中间层输出
                text_embedding, rec_embedding = model.get_embedding(batch)
                text_embeddings.append(text_embedding.cpu().numpy())
                rec_embeddings.append(rec_embedding.cpu().numpy())

        # 将所有 batch 的特征合并
        all_text_features = np.vstack(text_embeddings)
        all_rec_features = np.vstack(rec_embeddings)
        labels = np.concatenate(labels)
        return all_text_features, all_rec_features, labels

    def tsne_visualization(all_text_features, all_rec_features, labels, num_classes=2, perplexity=50):
        """
        进行 t-SNE 降维和可视化，将文本特征和推荐特征画在同一张图上，颜色区分特征类型（文本/推荐）。

        参数：
        - all_text_features: 文本特征数组
        - all_rec_features: 推荐系统特征数组
        - num_classes: 类别数量，默认为 2
        - perplexity: t-SNE 的 perplexity 参数，默认为 50
        """
        # t-SNE 降维代码保持不变
        tsne = TSNE(n_components=2, random_state=42)
        features_2d_text = tsne.fit_transform(all_text_features)
        features_2d_rec = tsne.fit_transform(all_rec_features)

        labels_text = np.zeros(all_text_features.shape[0])
        labels_rec = np.ones(all_rec_features.shape[0])

        features_2d = np.vstack((features_2d_text, features_2d_rec))
        labels = np.hstack((labels_text, labels_rec))

        # 设置图形样式
        plt.figure(figsize=(12, 10), dpi=300)

        # 设置背景透明
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['figure.facecolor'] = 'none'

        # 使用更优雅的颜色方案
        palette = sns.color_palette(["#e74d4a","#4b65e3"])

        # 绘制散点图，调整点的大小和透明度
        scatter_text = plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1],
                                   c=[palette[1]], label='text feature', alpha=0.7, s=50)
        scatter_rec = plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1],
                                  c=[palette[0]], label='rec feature', alpha=0.7, s=50)

        # 添加标题和标签
        plt.title(f"t-SNE  - {ablation_name}", fontsize=20, fontweight='bold')
        plt.xlabel("t-SNE x", fontsize=14)
        plt.ylabel("t-SNE y", fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        # 添加图例
        plt.legend(fontsize=16)

        # 保留坐标轴框
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.show()

        # 保存图形
        # save_path = './pics'
        # os.makedirs(save_path, exist_ok=True)
        # plt.savefig(os.path.join(save_path, f'tsne_{ablation_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight')

    # 调用函数
    all_text_features, all_rec_features, labels = extract_features(model, test_loader, accelerator.device)
    tsne_visualization(all_text_features, all_rec_features, labels, num_classes=2)  # 假设有2个类别


if __name__ == '__main__':
    cfg = create_parser()

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)
    set_seed(cfg.seed)
    logger.debug("This log will be printed in the main process only")
    accelerator.print(f'device {str(accelerator.device)} is used!')
    main(cfg)
    # CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 41011 --num_processes 2 bert_embedding_ddp.py
    # CUDA_VISIBLE_DEVICES=1 python pretrain_bertCTR_ddp.py
    # fix seed
