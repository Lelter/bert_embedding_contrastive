import csv
import os
import warnings

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn import BCEWithLogitsLoss

from config_finetune import create_parser
from datasets import BertCTRDataset
from models.LLM_CTR_finetune_model import bertCTRModel
from preprocessing.inputs import SparseFeat

warnings.filterwarnings("ignore")
import datetime
import json
import logging

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

from utils import EarlyStopping


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


def make_train_test_dfs(struct_data_path, text_data_path):
    # 划分训练集和测试集
    length = 4000

    # 读取结构化数据和文本数据
    struct_data = pd.read_csv(struct_data_path)
    # struct_data = pd.read_csv(struct_data_path)[:length]
    text_data = pd.read_csv(text_data_path)
    # text_data = pd.read_csv(text_data_path)[:length]
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
        cfg.text_path
    )
    sparse_feature_columns, sparse_all_feature_columns, train_struct_input, val_struct_input, test_struct_input = \
        process_struct_data(data_type, train_struct, val_struct, test_struct, struct_data)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    train_loader = build_loaders(train_struct_input, train_text,
                                 sparse_feature_columns, sparse_all_feature_columns, tokenizer, True)
    test_loader = build_loaders(test_struct_input, test_text,
                                sparse_feature_columns, sparse_all_feature_columns, tokenizer, False)
    val_loader = build_loaders(val_struct_input, val_text,
                               sparse_feature_columns, sparse_all_feature_columns, tokenizer, False)

    with open(cfg.meta_path) as fh:
        meta_data = json.load(fh)
    total_feature_num = meta_data['feature_num']
    model = bertCTRModel(cfg,
                         text_encoder_model=cfg.text_encoder_model,
                         sparse_feature_columns=sparse_feature_columns,
                         sparse_all_feature_columns=sparse_all_feature_columns,
                         struct_feature_num=total_feature_num,
                         )
    # optimizer = optim.Adagrad(model.parameters(), lr=cfg.lr,eps=1e-4)
    optimizer_dict = {
        'Adam': optim.Adam,
        'Adagrad': optim.Adagrad,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'AdamW': optim.AdamW
    }

    # 检查并获取优化器类型
    if cfg.optimizer not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    # 定义优化器的默认参数，添加参数只需修改此处
    optimizer_params = [
        {'params': model.rec_encoder.parameters(), 'lr': cfg.lr1, },
        {'params': model.text_encoder.parameters(), 'lr': cfg.lr2, }
    ]

    # 处理momentum参数，对于SGD和RMSprop才需要
    extra_params = {}
    if cfg.optimizer in ['SGD', 'RMSprop'] and hasattr(cfg, 'momentum'):
        extra_params['momentum'] = cfg.momentum

    # 初始化优化器
    optimizer = optimizer_dict[cfg.optimizer](optimizer_params, **extra_params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=cfg.step_size,gamma=cfg.gamma)

    # ======================================================================
    # initialize accelerator and auto move data/model to accelerator.device
    # model.load_state_dict(torch.load(f'ckpts/ablation/{cfg.dataset}/{cfg.llm}/final_paper.pth'), strict=False)
    model.load_state_dict(torch.load(f'ckpts/ablation/{cfg.dataset}/{cfg.llm}/final_{cfg.embedding_dim}.pth'), strict=False)
    no_grad_params = {'text_encoder.model.pooler.dense.weight',
                      'text_encoder.model.pooler.dense.bias'}  # 冻结 accelerate需要有梯度
    for name, param in model.named_parameters():
        if name in no_grad_params:
            param.requires_grad = False
    # Send everything through `accelerator.prepare`
    train_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, model, optimizer, scheduler
    )
    auc_list = []
    loss_list = []
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, disable=(not accelerator.is_local_main_process), ncols=100)
        
        for batch in pbar:
            optimizer.zero_grad()
            
            label = batch['label'].float().to(accelerator.device)
            output = model(batch).squeeze(1)
            
            loss = F.binary_cross_entropy_with_logits(output, label)
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_description(
                f"epoch {epoch + 1}: loss {loss.item():.5f}, "
                f"lr1 {optimizer.param_groups[0]['lr']:.5f}, "
                f"lr2 {optimizer.param_groups[1]['lr']:.5f}"
            )
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        val_logloss, val_auc = validate(test_loader, model)
        
        auc_list.append(val_auc)
        loss_list.append(val_logloss)
        early_stopping(val_auc)
        if early_stopping.early_stop:
            print("提前停止")
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        log_results(model, min(loss_list), max(auc_list), cfg)


def validate(val_loader, model):
    model.eval()
    with torch.no_grad():
        label_list = []
        pres_list = []
        for batch in val_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            label = batch['label'].float()
            output = torch.sigmoid(model(batch)).squeeze(1)
            label_list.append(label.cpu().numpy())
            pres_list.append(output.cpu().numpy())

    label_array = np.concatenate(label_list, axis=0)
    pres_array = np.concatenate(pres_list, axis=0)
    pres_array = np.expand_dims(pres_array, axis=1)
    label_array = np.expand_dims(label_array, axis=1)

    logloss = round(log_loss(label_array, pres_array), 4)
    auc = round(roc_auc_score(label_array, pres_array), 4)

    if accelerator.is_local_main_process:
        print(f"Val LogLoss: {logloss}, Val AUC: {auc}")

    return logloss, auc


def test(test_loader, model):
    model.eval()
    label_list, pres_list = [], []

    if accelerator.is_local_main_process:
        with torch.no_grad():
            for batch in test_loader:
                batch = {key: value.to(accelerator.device) for key, value in batch.items()}
                label = batch['label'].float()
                output = model(batch, "test").squeeze(1)

                label_list.append(label.cpu().numpy())
                pres_list.append(output.cpu().numpy())

        label_list = np.concatenate(label_list)
        pres_list = np.concatenate(pres_list)

        logloss = round(log_loss(label_list, pres_list), 6)
        auc = round(roc_auc_score(label_list, pres_list), 6)

        print(f"Test LogLoss: {logloss}, Test AUC: {auc}")
        log_results(model, logloss, auc, cfg)

    return logloss, auc


def log_results(model, logloss, auc, cfg):
    model_name = model.__class__.__name__
    file_path = f'./baseline_results/ablation/{cfg.backbone}/{cfg.llm}_{cfg.dataset}_{cfg.embedding_dim}.csv'

    # 检查目录是否存在，如果不存在则创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    headers = [
        "时间戳", "模型名称", "LLM", "骨干网络", "训练轮数", "批次大小",
        "学习率", "Dropout", "可训练", "alpha", "beta", "LR1", "LR2",
        "优化器", "AUC", "对数损失", "描述"
    ]

    row_data = [
        datetime.datetime.now().isoformat(),
        model_name,
        cfg.llm,
        cfg.backbone,
        cfg.epochs,
        cfg.batch_size,
        cfg.lr,
        cfg.dropout,
        cfg.trainable,
        cfg.alpha,
        cfg.beta,
        cfg.lr1,
        cfg.lr2,
        cfg.optimizer,
        auc,
        logloss,
        f"{cfg.alpha}_{cfg.beta}_{cfg.lr1}_{cfg.lr2}"
    ]

    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row_data)

    print(f"结果已记录到 {file_path}")


if __name__ == '__main__':
    cfg = create_parser()

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    if accelerator.is_local_main_process:
        print(json.dumps(vars(cfg), indent=2))
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)
    set_seed(cfg.seed)
    main(cfg)
    # CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 41011 --num_processes 2 bert_embedding_e2e_ddp.py
    # CUDA_VISIBLE_DEVICES=1 python pretrain_bertCTR_ddp.py
    # fix seed
