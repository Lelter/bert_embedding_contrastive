from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.modules_pretrain_modal import (RecEncoder, RecEncoder_AutoInt,
                                           RecEncoder_DCN, RecEncoder_DCNv2,
                                           RecEncoder_DeepFM, RecEncoder_PNN,
                                           RecEncoder_xDeepFM, TextEncoder)


class bertCTRModel(nn.Module):
    def __init__(
            self,
            cfg,
            text_embedding_dim, text_encoder_model,
            pretrained=True,
            sparse_feature_columns=None,
            sparse_all_feature_columns=None,
            struct_feature_num=None, mode="train"):
        super(bertCTRModel, self).__init__()
        embedding_dim = cfg.embedding_dim
        backbone_map: Dict[str, Type[nn.Module]] = {
            "DeepFM": RecEncoder_DeepFM,
            "DCNv2": RecEncoder_DCNv2,
            "PNN": RecEncoder_PNN,
            "AutoInt": RecEncoder_AutoInt,
            "xDeepFM": RecEncoder_xDeepFM,
            "DCN": RecEncoder_DCN,
            "Recoder": RecEncoder
        }

        if cfg.backbone in backbone_map:
            self.rec_encoder = backbone_map[cfg.backbone](
                sparse_feature_columns, sparse_all_feature_columns, struct_feature_num, cfg.t3
            )
            self.rec_encoder.forward = self.rec_encoder.modal_forward
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        self.text_encoder = TextEncoder(text_encoder_model, pretrained, cfg.trainable, cfg.max_length, embedding_dim)

        if cfg.llm == "SFR":
            self.text_encoder.forward = self.text_encoder.SFR_forward
        device = self.rec_encoder.device
        # self.t = nn.Parameter(torch.tensor(0.2, device=device), requires_grad=True)
        self.t1 = cfg.t1
        self.t2 = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=True)
        self.t3 = cfg.t3
        # self.a = 1
        # self.b = 1
        # self.c = 1

    #
    def FECM_loss(self, text_features_list1, text_features_list2):
        device = text_features_list1[0].device
        num_domains = len(text_features_list1)
        batch_size = text_features_list1[0].shape[0]

        # 初始化损失
        infoNCE = torch.tensor(0.0, device=device, requires_grad=True)

        for i in range(num_domains):
            # 计算正样本对的相似度
            sim_pos = F.cosine_similarity(text_features_list1[i], text_features_list2[i], dim=1)

            # 计算负样本对相似度
            sim_neg = F.cosine_similarity(text_features_list1[i].unsqueeze(1), text_features_list1[i].unsqueeze(0),
                                          dim=-1)

            # 计算损失
            pos_exp = torch.exp(sim_pos / self.t1).sum()
            neg_exp = torch.exp(sim_neg / self.t1).sum() - torch.exp(torch.diag(sim_neg) / self.t1).sum()

            # 对每个域的损失除以 batch size
            infoNCE = infoNCE - torch.log(pos_exp / neg_exp) / batch_size

        # 返回平均损失（除以域的数量）
        return infoNCE / num_domains

    def FDCM_loss(self, text_features_list1):
        device = text_features_list1[0].device

        # 计算每个域的embedding平均值作为质心
        embeddings_mean_list = [torch.mean(tensor, dim=0) for tensor in text_features_list1]
        result = torch.stack(embeddings_mean_list, dim=0)

        # 计算质心相似度attd
        attd = F.cosine_similarity(result.unsqueeze(1), result.unsqueeze(0), dim=-1)
        attd_softmax = F.softmax(attd / self.t2, dim=1)  # 按行计算softmax,添加温度参数

        # 初始化相似度矩阵
        num_features = len(text_features_list1)
        sd_matrix = torch.zeros((num_features, num_features)).to(device)

        # 计算域内和域间相似度
        for i in range(num_features):
            sim_pos1 = F.cosine_similarity(text_features_list1[i].unsqueeze(1), text_features_list1[i].unsqueeze(0),
                                           dim=-1)
            # sim_pos1 = torch.exp(sim_pos1 / self.t2)
            upper_triangular = torch.triu(sim_pos1, diagonal=1)  # 已经除以batchsize*batchsize
            mean_value = torch.mean(upper_triangular[upper_triangular != 0])
            sd_matrix[i, i] = mean_value

            for j in range(i + 1, num_features):
                sim_pos2 = F.cosine_similarity(text_features_list1[i].unsqueeze(1), text_features_list1[j].unsqueeze(0),
                                               dim=-1)
                # sim_pos2 = torch.exp(sim_pos2 / self.t2)
                st_outer = torch.mean(sim_pos2)  # 已经除以batchsize*batchsize
                sd_matrix[i, j] = st_outer
                sd_matrix[j, i] = st_outer

        # 计算softmax后的相似度矩阵
        sd_matrix_softmax = F.softmax(sd_matrix / self.t2, dim=1)

        attd_labels = attd_softmax.view(-1).long()
        sd_outputs = sd_matrix_softmax.view(-1)
        sd_outputs = torch.log(sd_outputs)
        # loss = F.cross_entropy(sd_outputs, attd_labels)
        # KL散度
        # loss = F.kl_div(torch.log(sd_outputs), attd_labels, reduction='batchmean')
        loss = torch.nn.NLLLoss()(sd_outputs, attd_labels)
        return loss

    def forward(self, batch, mode="train"):
        #

        text_features_list1 = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_features_list2 = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Compute losses
        FECM_loss = self.FECM_loss(text_features_list1, text_features_list2)  # instance
        # domain_cross_loss = self.domain_cross_loss_once(text_features_list1)
        FDCM_loss = self.FDCM_loss(text_features_list1)  # domain
        # Compute logits
        MCM_loss = self.rec_encoder(text_features_list1, batch['rec_data'])  # modal
        # Compute total loss
        total_loss = [100*FECM_loss, FDCM_loss, MCM_loss]
        return total_loss

    def get_embedding(self, batch):
        text_features_list = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embedding, rec_embedding = self.rec_encoder.get_embedding(text_features_list, batch['rec_data'])
        return text_embedding, rec_embedding
