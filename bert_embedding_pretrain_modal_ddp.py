import csv
import os
import warnings

from sklearn.preprocessing import LabelEncoder
from torch import optim

from config_pretrain_modal import create_parser
from datasets import BertCTRDataset
from models.LLM_CTR_modal_model import bertCTRModel
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
from transformers import AutoTokenizer

# from model.MaskCTR_ddp import MaskCTR
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


def make_train_test_dfs(struct_data_path, text_data_path, seed):
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
                                sparse_feature_columns, sparse_all_feature_columns, tokenizer, False)
    val_loader = build_loaders(val_struct_input, val_text,
                               sparse_feature_columns, sparse_all_feature_columns, tokenizer, False)
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
    # optimizer_params = [
    #     {'params': model.rec_encoder.parameters(), 'lr': cfg.lr1},
    #     {'params': model.text_encoder.parameters(), 'lr': cfg.lr2}
    # ]
    optimizer_params = [
        {'params': model.parameters(), 'lr': cfg.lr1},
    ]

    # 处理momentum参数，对于SGD和RMSprop才需要
    extra_params = {}
    if cfg.optimizer in ['SGD', 'RMSprop'] and hasattr(cfg, 'momentum'):
        extra_params['momentum'] = cfg.momentum

    # 初始化优化器
    optimizer = optimizer_dict[cfg.optimizer](optimizer_params, **extra_params)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=cfg.step_size,gamma=cfg.gamma)

    # ======================================================================
    # initialize accelerator and auto move data/model to accelerator.device

    no_grad_params = {'text_encoder.model.pooler.dense.weight',
                      'text_encoder.model.pooler.dense.bias'}  # 冻结 accelerate需要有梯度
    for name, param in model.named_parameters():
        if name in no_grad_params:
            param.requires_grad = False
    # Send everything through `accelerator.prepare`
    train_loader, model, optimizer, = accelerator.prepare(
        train_loader, model, optimizer,
    )

    best_loss = float('inf')
    patience_counter = 0
    t2_list = []

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(not accelerator.is_local_main_process),
                    ncols=100)
        train_loss = 0.0
        for batch_idx, batch in pbar:
            optimizer.zero_grad()
            loss_list,t2 = model(batch, "train")
            t2_list.append(t2.item())
            loss = sum(x for x in loss_list)
            accelerator.backward(loss)
            optimizer.step()

            train_loss += loss.item()
            # Ensure loss_list has at least three elements and each element is a tensor
            if len(loss_list) >= 3 and all(isinstance(loss, torch.Tensor) for loss in loss_list):
                pbar.set_description(
                    f"epoch {epoch + 1}: loss0 {loss_list[0].item():.5f}, loss1 {loss_list[1].item():.5f}, loss2 {loss_list[2].item():.5f} t2 {t2.item():.5f}")
            else:
                pbar.set_description(f"epoch {epoch + 1}: loss {loss.item():.5f}")

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"轮次 {epoch + 1} 平均训练损失: {avg_train_loss:.5f}")

        # Check for loss convergence
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            logger.info(f"训练提前停止于轮次 {epoch + 1}，因为损失不再下降")
            break

    accelerator.wait_for_everyone()
    unwrap_model = accelerator.unwrap_model(model)
    save_dir = f'ckpts/ablation/{cfg.dataset}/{cfg.llm}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(unwrap_model.state_dict(), os.path.join(save_dir, f'final_{cfg.embedding_dim}.pth'))
    np.save(os.path.join(save_dir, f't2.npy'),t2_list)


def validate(val_loader, model):
    model.eval()
    label_list, pres_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}

            label = batch['label'].float()
            output = model(batch, "test")[0].squeeze(1)

            label_list.append(label)
            pres_list.append(output)

    label_tensor = torch.cat(label_list)
    pres_tensor = torch.cat(pres_list)

    label_numpy = label_tensor.cpu().numpy()
    pres_numpy = pres_tensor.cpu().numpy()

    label_numpy = np.expand_dims(label_numpy, axis=1)
    pres_numpy = np.expand_dims(pres_numpy, axis=1)

    logloss = round(log_loss(label_numpy, pres_numpy), 4)
    auc = round(roc_auc_score(label_numpy, pres_numpy), 4)

    if accelerator.is_local_main_process:
        logger.info(f"验证 LogLoss: {logloss}, 验证 AUC: {auc}")

    return logloss, auc


def test(test_loader, model):
    model.eval()
    label_list, pres_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            label = batch['label'].float()
            output = model(batch, "test")[0].squeeze(1)

            label_list.append(label)
            pres_list.append(output)

    label_tensor = torch.cat(label_list)
    pres_tensor = torch.cat(pres_list)

    label_numpy = label_tensor.cpu().numpy()
    pres_numpy = pres_tensor.cpu().numpy()

    logloss = round(log_loss(label_numpy, pres_numpy), 6)
    auc = round(roc_auc_score(label_numpy, pres_numpy), 6)

    if accelerator.is_local_main_process:
        logger.info(f"测试 LogLoss: {logloss}, 测试 AUC: {auc}")
        log_results(model, logloss, auc)

    return logloss, auc


def log_results(model, logloss, auc):
    model_name = model.__class__.__name__
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_data = {
        "时间戳": current_time,
        "模型名称": model_name,
        "LLM": cfg.llm,
        "骨干网络": cfg.backbone,
        "训练轮数": cfg.epochs,
        "批次大小": cfg.batch_size,
        "学习率": cfg.lr,
        "Dropout": cfg.dropout,
        "可训练": cfg.trainable,
        "温度": cfg.t,
        "学习率1": cfg.lr1,
        "学习率2": cfg.lr2,
        "优化器": cfg.optimizer,
        "AUC": auc,
        "LogLoss": logloss,
        "描述": ""
    }

    file_path = f'./baseline_results/ablation/{cfg.dataset}_{model_name}.csv'

    df = pd.DataFrame([result_data])

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False, mode='w')
    else:
        df.to_csv(file_path, index=False, mode='a', header=False)

    logger.info(f"结果已保存到 {file_path}")


if __name__ == '__main__':
    cfg = create_parser()

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)
    set_seed(cfg.seed)
    accelerator.print(f'device {str(accelerator.device)} is used!')
    main(cfg)
    # CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 41011 --num_processes 2 bert_embedding_ddp.py
    # CUDA_VISIBLE_DEVICES=1 python pretrain_bertCTR_ddp.py
    # fix seed
