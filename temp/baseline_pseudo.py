import csv
import os
import warnings
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch import optim
from models.LLM_CTR_model import bertCTRModel
from datasets import BertCTRDataset
from preprocessing.inputs import SparseFeat
from config_e2e import create_parser
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

import random
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
# from model.MaskCTR_ddp import MaskCTR
from utils import AvgMeter, get_lr, MetricLogger, SmoothedValue, is_main_process, get_rank, EarlyStopping
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
from accelerate import Accelerator
from accelerate.utils import set_seed
import datetime
import json
from accelerate.logging import get_logger
import logging
from sklearn.metrics import precision_score, recall_score, f1_score


def process_struct_data(data_source, train, val, test, data):
    # 对结构化数据进行处理
    embedding_dim = cfg.embedding_dim
    if data_source == 'movielens':
        sparse_features = ['user_id', 'gender', 'age', 'occupation', 'zip', 'movie_id', 'title', 'genres']
    elif data_source == 'bookcrossing':
        sparse_features = ['User ID', 'Location', 'Age', 'ISBN', 'Book title', 'Author', 'Publication year',
                           'Publisher']
    elif data_source == 'goodreads':
        sparse_features = ['User ID', 'Book ID', 'Book title', 'Book genres', 'Average rating',
                           'Number of book reviews', 'Author ID', 'Author name',
                           'Number of pages', 'eBook flag', 'Format', 'Publisher', 'Publication year', 'Work ID',
                           'Media type']

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                              for i, feat in enumerate(sparse_features)]
    label_encoders = {feat: LabelEncoder().fit(data[feat]) for feat in sparse_features}
    for feat in sparse_features:
        # data[feat] = lbe.fit_transform(data[feat])
        train[feat] = label_encoders[feat].transform(train[feat])
        test[feat] = label_encoders[feat].transform(test[feat])
        val[feat] = label_encoders[feat].transform(val[feat])
    linear_feature_columns = sparse_feature_columns
    dnn_feature_columns = sparse_feature_columns

    train_model_input = {name: train[name] for name in sparse_features}
    test_model_input = {name: test[name] for name in sparse_features}
    val_model_input = {name: val[name] for name in sparse_features}

    return linear_feature_columns, dnn_feature_columns, train_model_input, val_model_input, test_model_input


def make_train_test_dfs(struct_data_path, text_data_path, seed):
    # 划分训练集和测试集
    length = 40000

    # 读取结构化数据和文本数据
    struct_data = pd.read_csv(struct_data_path)
    text_data = pd.read_csv(text_data_path, )
    text_data = text_data.sample(frac=1)
    # struct_data = struct_data[:length]
    text_data = text_data[:length]
    # text_data['label'] = struct_data['label']
    # 划分数据集：80% 训练集，10% 验证集，10% 测试集
    train_size = int(len(text_data) * 0.8)
    val_size = int(len(text_data) * 0.1)

    # train_struct = struct_data.iloc[:train_size].copy()
    # val_struct = struct_data.iloc[train_size:train_size + val_size].copy()
    # test_struct = struct_data.iloc[train_size + val_size:].copy()

    train_text = text_data.iloc[:train_size].copy()
    val_text = text_data.iloc[train_size:train_size + val_size].copy()
    test_text = text_data.iloc[train_size + val_size:].copy()

    return train_text, val_text, test_text, struct_data


def build_loaders(text_input, tokenizer, ):
    dataset = BertCTRDataset(
        text_input["rephrase"].values,
        text_input["pseudo_label"].values,

        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
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
    train_text, val_text, test_text, struct_data = make_train_test_dfs(
        cfg.struct_path,
        cfg.text_path, cfg.seed,
    )
    # linear_feature_columns, dnn_feature_columns, train_struct_input, val_struct_input, test_struct_input = \
    #     process_struct_data(data_type, train_struct, val_struct, test_struct, struct_data)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    train_loader = build_loaders(train_text,
                                 tokenizer)
    test_loader = build_loaders(test_text,
                                tokenizer)
    val_loader = build_loaders(val_text,
                               tokenizer)
    with open(cfg.meta_path) as fh:
        meta_data = json.load(fh)
    total_feature_num = meta_data['feature_num']
    model = bertCTRModel(cfg, rec_embedding_dim=cfg.rec_embedding_dim, text_embedding_dim=cfg.text_embedding_dim,
                         text_encoder_model=cfg.text_encoder_model,
                         struct_feature_num=total_feature_num,
                         )
    optimizer = optim.Adagrad(model.parameters(), lr=cfg.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # ======================================================================
    # initialize accelerator and auto move data/model to accelerator.device

    no_grad_params = {'text_encoder.model.pooler.dense.weight',
                      'text_encoder.model.pooler.dense.bias'}  # 冻结 accelerate需要有梯度
    for name, param in model.named_parameters():
        if name in no_grad_params:
            param.requires_grad = False
    # Send everything through `accelerator.prepare`
    train_loader, model, optimizer = accelerator.prepare(
        train_loader, model, optimizer
    )
    # model.find_unused_parameters = True
    # ======================================================================

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(not accelerator.is_local_main_process))
        total_loss = 0

        for batch_idx, (batch) in pbar:
            optimizer.zero_grad()

            label = batch['label'].long().to(accelerator.device)
            output = model(batch).squeeze(1)

            # 单标签多分类
            loss = F.cross_entropy(output, label, reduction='sum')

            accelerator.backward(loss)
            # print_gradients(model)

            optimizer.step()

            pbar.set_description(f"epoch {epoch + 1} : train loss {loss.item():.5f}")
            if accelerator.is_local_main_process:
                total_loss += loss.item()
        # 常规评估
        # if accelerator.is_local_main_process:
        p, r, f = validate(val_loader, model)  # 验证
        early_stopping(f)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # ======================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        p, r, f = test(test_loader, model)  # 测试
        unwrap_model = accelerator.unwrap_model(model)
        unwrap_optim = accelerator.unwrap_model(optimizer)
        save_path = cfg.load_prefix_path + cfg.ckpt_dir

        # torch.save({
        #     'model_state': unwrap_model.state_dict(),
        #     'optim_state': unwrap_optim.state_dict()},
        #     save_path + str(cfg.lr) + "_" + str(cfg.dropout) + "_" + str(cfg.epochs) + "_" + str(
        #         cfg.backbone) + "_" + str(cfg.llm) + "_" + str(cfg.embedding_dim) + "_" + str(logloss) + "_" + str(
        #         auc) + ".pt")
        # logger.info(f'checkpoint is saved...')
        # ======================================================================

        # logger.info(f'checkpoint ckpt_{epoch + 1}.pt is saved...')
        # net_dict = accelerator.get_state_dict(model)
        # accelerator.save(net_dict,
        #                  save_path + str(cfg.lr) + "_" + str(cfg.dropout) + "_" + str(cfg.epochs) + "_" + str(
        #                      cfg.backbone) + "_" + str(cfg.llm) + "_" + str(cfg.embedding_dim))
        # with open(f'baseline_results/{data_source}_{model_name}.txt', 'a+') as writer:
        #     writer.write(' '.join(writer_text) + '\n')

        # accelerator.print(f"epoch【{epoch}】@{nowtime} --> eval_accuracy= {eval_metric:.2f}%")
        # net_dict = accelerator.get_state_dict(model)
        # accelerator.save(net_dict, ckpt_path + "_" + str(epoch))
        # # ======================================================================

        pass


def validate(val_loader, model):
    model.eval()
    p_list = []
    r_list = []
    f_list = []
    with torch.no_grad():
        for batch_idx, (batch) in enumerate(val_loader):
            for key in batch.keys():
                # if key not in ['label']:
                batch[key] = batch[key].to(accelerator.device)
            label = batch['label'].long()
            output = model(batch).squeeze(1).to(accelerator.device)
            # 计算precison、recall、f1
            _, predicted = torch.max(output, 1)

            # 将预测和实际标签从 GPU 转回 CPU (如果需要)
            predicted = predicted.cpu()
            label = label.cpu()

            # 计算 precision, recall, f1-score
            precision = precision_score(label, predicted, average='weighted')
            recall = recall_score(label, predicted, average='weighted')
            f1 = f1_score(label, predicted, average='weighted')
            p_list.append(precision)
            r_list.append(recall)
            f_list.append(f1)
    if accelerator.is_local_main_process:
        print(f"{sum(p_list) / len(p_list)}, {sum(r_list) / len(r_list)}, {sum(f_list) / len(f_list)}")
    return sum(p_list) / len(p_list), sum(r_list) / len(r_list), sum(f_list) / len(f_list)


def test(test_loader, model):
    model.eval()
    p_list = []
    r_list = []
    f_list = []

    if accelerator.is_local_main_process:
        with torch.no_grad():
            for batch_idx, (batch) in enumerate(test_loader):
                for key in batch.keys():
                    # if key not in ['label']:
                    batch[key] = batch[key].to(accelerator.device)
                label = batch['label'].long()
                output = model(batch).squeeze(1).to(accelerator.device)
                # 计算precison、recall、f1
                _, predicted = torch.max(output, 1)

                # 将预测和实际标签从 GPU 转回 CPU (如果需要)
                predicted = predicted.cpu()
                label = label.cpu()

                # 计算 precision, recall, f1-score
                precision = precision_score(label, predicted, average='weighted')
                recall = recall_score(label, predicted, average='weighted')
                f1 = f1_score(label, predicted, average='weighted')
                p_list.append(precision)
                r_list.append(recall)
                f_list.append(f1)
        print(f"{sum(p_list) / len(p_list)}, {sum(r_list) / len(r_list)}, {sum(f_list) / len(f_list)}")
    return sum(p_list) / len(p_list), sum(r_list) / len(r_list), sum(f_list) / len(f_list)

    # writer.write(' '.join(writer_text) + '\n')


if __name__ == '__main__':
    cfg = create_parser()
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)
    set_seed(cfg.seed)
    logger.debug("This log will be printed in the main process only")
    accelerator.print(f'device {str(accelerator.device)} is used!')
    main(cfg)
    # CUDA_VISIBLE_DEVICES=1,0 accelerate launch --main_process_port 41011 --num_processes 2 bert_embedding_e2e_ddp.py
    # CUDA_VISIBLE_DEVICES=1 python pretrain_bertCTR_ddp.py
    # fix seed
