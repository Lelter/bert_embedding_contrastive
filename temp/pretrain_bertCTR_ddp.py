import csv
import os
import warnings

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch import optim

from config_e2e import create_parser
from datasets import BertCTRDataset
from models.LLM_CTR_model import bertCTRModel
from preprocessing.inputs import SparseFeat

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

from transformers import AutoTokenizer
from tqdm import tqdm
# from model.MaskCTR_ddp import MaskCTR
from utils import EarlyStopping
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import datetime
import json
from accelerate.logging import get_logger
import logging


def process_struct_data(data_source, train, val, test, data):
    # 对结构化数据进行处理
    embedding_dim = cfg.embedding_dim
    if data_source == 'movielens':
        # sparse_features = ['user_id', 'gender', 'age', 'occupation', 'zip', 'movie_id', 'title', 'genres']
        sparse_features = [ 'gender', 'age', 'occupation', 'zip', 'title', 'genres']
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
    # text_data = text_data.sample(frac=1)
    # struct_data = struct_data[:length]
    # text_data = text_data[:length]
    text_data['label'] = struct_data['label']
    # 划分数据集：80% 训练集，10% 验证集，10% 测试集
    train_size = int(len(text_data) * 0.8)
    val_size = int(len(text_data) * 0.1)

    train_struct = struct_data.iloc[:train_size].copy()
    val_struct = struct_data.iloc[train_size:train_size + val_size].copy()
    test_struct = struct_data.iloc[train_size + val_size:].copy()

    train_text = text_data.iloc[:train_size].copy()
    val_text = text_data.iloc[train_size:train_size + val_size].copy()
    test_text = text_data.iloc[train_size + val_size:].copy()

    return train_struct, val_struct, test_struct, train_text, val_text, test_text, struct_data


def build_loaders(struct_input,text_input, linear_feature_columns, dnn_feature_columns, tokenizer,  ):
    dataset = BertCTRDataset(
        struct_input,
        text_input,
        text_input["label"].values,
        text_input['pseudo_label'].values,
        sparse_feature_columns=linear_feature_columns,
        sparse_all_feature_columns=dnn_feature_columns,
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
    train_struct, val_struct, test_struct, train_text, val_text, test_text, struct_data = make_train_test_dfs(
        cfg.struct_path,
        cfg.text_path, cfg.seed,
    )
    linear_feature_columns, dnn_feature_columns, train_struct_input, val_struct_input, test_struct_input = \
        process_struct_data(data_type, train_struct, val_struct, test_struct, struct_data)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    train_loader = build_loaders(train_struct_input, train_text,
                                 linear_feature_columns, dnn_feature_columns, tokenizer)
    test_loader = build_loaders(test_struct_input, test_text,
                                linear_feature_columns, dnn_feature_columns, tokenizer)
    val_loader = build_loaders(val_struct_input, val_text,
                               linear_feature_columns, dnn_feature_columns, tokenizer)
    with open(cfg.meta_path) as fh:
        meta_data = json.load(fh)
    total_feature_num = meta_data['feature_num']
    model = bertCTRModel(cfg, rec_embedding_dim=cfg.rec_embedding_dim, text_embedding_dim=cfg.text_embedding_dim,
                         text_encoder_model=cfg.text_encoder_model,
                         struct_linear_feature_columns=linear_feature_columns,
                         struct_dnn_feature_columns=dnn_feature_columns,
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
    # for batch_idx, last_batch in enumerate(train_loader):
    #     pass
    # output=model(last_batch).sqeeze(1)
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(not accelerator.is_local_main_process))


        total_loss = 0

        for batch_idx, (batch) in pbar:
            optimizer.zero_grad()

            label = batch['label'].float().to(accelerator.device)
            output,loss2 = model(batch)
            output=output.squeeze(1)
            # 单标签多分类
            loss1 = F.binary_cross_entropy(output, label)
            loss=loss1+loss2
            accelerator.backward(loss)
            # print_gradients(model)

            optimizer.step()

            pbar.set_description(f"epoch {epoch + 1} : train loss1 {loss1.item():.5f},contrastive loss {loss2.item():.5f}")
            if accelerator.is_local_main_process:
                total_loss += loss.item()
        # 常规评估
        # if accelerator.is_local_main_process:
        val_logloss, val_auc = validate(test_loader, model)  # 验证
        early_stopping(val_auc)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # ======================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        logloss, auc = test(test_loader, model)  # 测试
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

        pass


def validate(val_loader, model):
    model.eval()
    label_list = []
    pres_list = []
    with torch.no_grad():
        for batch_idx, (batch) in enumerate(val_loader):
            for key in batch.keys():
                # if key not in ['label']:
                batch[key] = batch[key].to(accelerator.device)
            label = batch['label'].float()
            output = model(batch)[0].squeeze(1).to(accelerator.device)
            label_list.append(label.cpu().detach().numpy())
            pres_list.append(output.cpu().detach().numpy())

    label_list = np.concatenate(label_list)
    pres_list = np.concatenate(pres_list)
    logloss = (round(log_loss(label_list, pres_list), 6))
    auc = (round(roc_auc_score(label_list, pres_list), 6))
    if accelerator.is_local_main_process:
        print(f"Val LogLoss: {logloss}, Val AUC: {auc}")
    return logloss, auc


def test(test_loader, model):
    model.eval()
    label_list = []
    pres_list = []
    if accelerator.is_local_main_process:
        with torch.no_grad():
            for batch_idx, (batch) in enumerate(test_loader):
                for key in batch.keys():
                    # if key not in ['label']:
                    batch[key] = batch[key].to(accelerator.device)
                label = batch['label'].float()
                output = model(batch)[0].squeeze(1).to(accelerator.device)
                label_list.append(label.cpu().detach().numpy())
                pres_list.append(output.cpu().detach().numpy())

        label_list = np.concatenate(label_list)
        pres_list = np.concatenate(pres_list)
        logloss = str(round(log_loss(label_list, pres_list), 6))
        auc = str(round(roc_auc_score(label_list, pres_list), 6))
        print(f"Test LogLoss: {logloss}, Test AUC: {auc}")
        model_name = model.__class__.__name__
        writer_text = [str(datetime.datetime.now()), model_name, str(cfg.llm), str(cfg.backbone), str(cfg.epochs),
                       str(cfg.batch_size), str(cfg.lr),
                       str(cfg.dropout), str(cfg.trainable), auc, logloss, " "]
        file_path = f'./baseline_results/{cfg.dataset}_{model_name}.csv'
        # 表头
        headers = ["Timestamp", "Model Name", "LLM", "Backbone", "Epochs", "Batch Size", "Learning Rate", "Dropout",
                   "AUC", "Logloss", "describe"]
        # 检查文件是否存在
        file_exists = os.path.isfile(file_path)
        # 打开文件并写入数据
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(headers)
            # 写入数据行
            writer.writerow(writer_text)
        # with open(f'baseline_results/{cfg.dataset}_{model_name}.txt', 'a+') as writer:
        #     writer.write(' '.join(writer_text) + '\n')
        # with open(f'./baseline_results/{cfg.dataset}_{model_name}.txt', 'a+') as writer:  # 打印key和value
        #     writer.write(' '.join(writer_text) + '\n')
        return logloss, auc


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
