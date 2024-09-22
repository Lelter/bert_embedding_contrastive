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
                sparse_feature_columns, sparse_all_feature_columns, struct_feature_num
            )
            self.rec_encoder.forward = self.rec_encoder.modal_forward
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        self.text_encoder = TextEncoder(text_encoder_model, pretrained, cfg.trainable, cfg.max_length, embedding_dim)

        if cfg.llm == "SFR":
            self.text_encoder.forward = self.text_encoder.SFR_forward
        device = self.rec_encoder.device
        # self.t = nn.Parameter(torch.tensor(0.2, device=device), requires_grad=True)
        self.t = cfg.t
        self.a = 1
        self.b = 1
        self.c = 1

    #
    def domain_inner_loss(self, text_features_list1, text_features_list2):
        device = text_features_list1[0].device
        num_domains = len(text_features_list1)

        # 初始化损失
        infoNCE = torch.tensor(0.0, device=device, requires_grad=True)

        for i in range(num_domains):
            # 计算正样本对的相似度，自己和自己为正样本对
            sim_pos = F.cosine_similarity(text_features_list1[i], text_features_list2[i], dim=1)

            # 计算负样本对相似度，自己和别人为负样本对
            sim_neg = F.cosine_similarity(text_features_list1[i].unsqueeze(1), text_features_list1[i].unsqueeze(0),
                                          dim=-1)

            # 计算损失
            pos_exp = torch.exp(sim_pos / self.t).sum()

            neg_exp = torch.exp(sim_neg / self.t).sum() - torch.exp(
                torch.diag(sim_neg) / self.t).sum()  # 除了对角线，其他的都是负样本
            infoNCE = infoNCE - torch.log(pos_exp / (pos_exp + neg_exp))

        # 返回平均损失
        return infoNCE / num_domains

    # def domain_cross_loss(self, text_features_list):
    #     device = text_features_list[0].device
    #     embeddings = text_features_list
    #     batch_size = embeddings[0].shape[0]
    #     keys = list(range(len(embeddings)))
    #     embeddings = {key: embeddings[key] for key in keys}  # 6,BS,dims
    #     # keys = list(embeddings.keys())
    #
    #     positive_loss = torch.tensor(0.0, requires_grad=True).to(device)
    #
    #     pos_distance_dict = {}
    #     neg_distance_dict = {key: torch.zeros(batch_size).to(device) for key in keys}
    #
    #     # 计算正样本对的相似度矩阵
    #     for key, embed in embeddings.items():
    #         pos_sim = F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=-1)
    #         pos_distance_dict[key] = torch.exp(pos_sim / self.t)
    #     # 负样本相似度矩阵
    #     for i in range(len(keys)):
    #         embed_i = embeddings[keys[i]]
    #         for j in range(i + 1, len(keys)):
    #             embed_j = embeddings[keys[j]]
    #             neg_sim = F.cosine_similarity(embed_i, embed_j, dim=1)
    #             neg_distances = torch.exp(neg_sim / self.t)
    #             neg_distance_dict[keys[i]] += neg_distances
    #             neg_distance_dict[keys[j]] += neg_distances
    #
    #     # 计算正样本和负样本对的InfoNCE损失
    #     for key, pos_distance in pos_distance_dict.items():
    #         for each_row in range(batch_size):  # 正样本对太多！
    #             if each_row == batch_size - 1:  # 最后一行没有正样本对
    #                 continue
    #             pos_dist_row = pos_distance[each_row, each_row + 1:]  # 每行正样本对的距离
    #             neg_dist_row = neg_distance_dict[key][each_row]  # 每行负样本对的距离和
    #
    #             infoNCE = torch.log(pos_dist_row / (pos_dist_row + neg_dist_row))
    #             positive_loss_each_row = infoNCE.sum() / (batch_size - each_row - 1)  # 除以正样本对的数量
    #
    #             positive_loss += positive_loss_each_row
    #             # print(positive_loss, each_row, key)
    #
    #     all_loss = -positive_loss / (batch_size * len(keys))
    #     return all_loss

    def domain_cross_loss(self, text_features_list):
        device = text_features_list[0].device
        embeddings = text_features_list
        batch_size = embeddings[0].shape[0]

        positive_loss = torch.tensor(0.0, requires_grad=True).to(device)

        # 计算正样本对的相似度矩阵
        pos_sim = [F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=-1) for embed in embeddings]
        pos_distance = [torch.exp(sim / self.t) for sim in pos_sim]

        # 负样本相似度矩阵
        neg_sim = torch.stack([F.cosine_similarity(embeddings[i].unsqueeze(1), embeddings[j].unsqueeze(0), dim=-1)
                               for i in range(len(embeddings)) for j in range(i + 1, len(embeddings))])
        neg_distances = torch.exp(neg_sim / self.t).sum(dim=1)

        # 计算InfoNCE损失
        for i, pos_dist in enumerate(pos_distance):
            for each_row in range(batch_size - 1):  # 最后一行没正样本对
                pos_dist_row = pos_dist[each_row, each_row + 1:]  # 获取每行的正样本对距离
                neg_dist_row = neg_distances[i][each_row]  # 获取每行负样本对的距离和

                infoNCE = torch.log(pos_dist_row / (pos_dist_row + neg_dist_row))
                positive_loss += infoNCE.sum() / (batch_size - each_row - 1)  # 除以正样本对数量

        all_loss = -positive_loss / (batch_size * len(embeddings))
        return all_loss

    def domain_cross_loss_once(self, text_features_list):
        embeddings = text_features_list
        batch_size = embeddings[0].shape[0]

        # 计算正样本对的相似度矩阵
        pos_sim = [F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=-1) for embed in embeddings]
        pos_distance = [torch.exp(sim / self.t) for sim in pos_sim]

        # 负样本相似度矩阵
        neg_sim = torch.cat(
            [F.cosine_similarity(embeddings[i].unsqueeze(1), embeddings[j].unsqueeze(0), dim=-1).unsqueeze(0)
             for i in range(len(embeddings)) for j in range(i + 1, len(embeddings))], dim=0)
        neg_distances = torch.exp(neg_sim / self.t)

        # 计算InfoNCE损失
        positive_loss = 0.0  # 初始化为标量
        for i, pos_dist in enumerate(pos_distance):
            pos_dist_upper = torch.triu(pos_dist, diagonal=1)  # 获取上三角矩阵（不包括对角线）
            neg_dist_sum = neg_distances[i].sum(dim=0)  # 计算每列负样本距离和
            pos_dist_sum = pos_dist_upper.sum()
            infoNCE = torch.log(pos_dist_sum / (pos_dist_sum + neg_dist_sum))
            positive_loss += infoNCE.sum()  # 除以正样本对数量

        all_loss = -positive_loss / len(embeddings)
        return all_loss

    def domain_in_cross_regulation_loss(self, text_features_list1):
        device = text_features_list1[0].device

        # 计算每个域的embedding平均值作为质心
        embeddings_mean_list = [torch.mean(tensor, dim=0) for tensor in text_features_list1]
        result = torch.stack(embeddings_mean_list, dim=0)

        # 计算质心相似度attd
        attd = F.cosine_similarity(result.unsqueeze(1), result.unsqueeze(0), dim=-1)
        attd_softmax = F.softmax(attd, dim=1)  # 按行计算softmax

        # 初始化相似度矩阵
        num_features = len(text_features_list1)
        sd_matrix = torch.zeros((num_features, num_features)).to(device)

        # 计算域内和域间相似度
        for i in range(num_features):
            sim_pos1 = F.cosine_similarity(text_features_list1[i].unsqueeze(1), text_features_list1[i].unsqueeze(0),
                                           dim=-1)
            sim_pos1 = torch.exp(sim_pos1 / self.t)
            upper_triangular = torch.triu(sim_pos1, diagonal=1)
            mean_value = torch.mean(upper_triangular[upper_triangular != 0])
            sd_matrix[i, i] = mean_value

            for j in range(i + 1, num_features):
                sim_pos2 = F.cosine_similarity(text_features_list1[i].unsqueeze(1), text_features_list1[j].unsqueeze(0),
                                               dim=-1)
                sim_pos2 = torch.exp(sim_pos2 / self.t)
                st_outer = torch.mean(sim_pos2)
                sd_matrix[i, j] = st_outer
                sd_matrix[j, i] = st_outer

        # 计算softmax后的相似度矩阵
        sd_matrix_softmax = F.softmax(sd_matrix, dim=1)

        attd_labels = attd_softmax.view(-1)
        sd_outputs = sd_matrix_softmax.view(-1)
        # loss = F.cross_entropy(sd_outputs, attd_labels)
        # KL散度
        loss = F.kl_div(torch.log(sd_outputs), attd_labels, reduction='batchmean')
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
        domain_inner_loss = self.domain_inner_loss(text_features_list1, text_features_list2)  # instance
        # domain_cross_loss = self.domain_cross_loss_once(text_features_list1)
        domain_in_cross_regulation_loss = self.domain_in_cross_regulation_loss(text_features_list1)  # domain
        # Compute logits
        modal_align_loss = self.rec_encoder(text_features_list1, batch['rec_data'])  # modal
        # Compute total loss
        # total_loss = domain_inner_loss*domain_inner_loss+domain_in_cross_regulation_loss* domain_in_cross_regulation_loss
        total_loss = [domain_inner_loss, domain_in_cross_regulation_loss,modal_align_loss ]
        return total_loss

    def get_embedding(self, batch):
        text_features_list = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embedding, rec_embedding = self.rec_encoder.get_embedding(text_features_list, batch['rec_data'])
        return text_embedding, rec_embedding
