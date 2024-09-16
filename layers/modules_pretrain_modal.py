import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn, Tensor
from transformers import AutoModel
from transformers import BertConfig, BertModel, DistilBertModel

from layers.core import DNN
from layers.core import concat_fun
from layers.interaction import CrossNet, CIN
from layers.interaction import CrossNetMix
from layers.interaction import FM
from layers.interaction import InnerProductLayer
from layers.interaction import InteractingLayer
from models.base_model import BaseModel, Linear
from preprocessing.inputs import combined_dnn_input
import torch.nn.functional as F


class TextEncoder(nn.Module):
    # pretrain=True
    def __init__(self, model_name, pretrained, trainable, max_length, embedding_dim):
        super().__init__()
        print('pretrained: ', pretrained)
        self.model = self.load_model(model_name, trainable)
        self.max_length = max_length
        self.linear = nn.Linear(self.model.config.hidden_size, embedding_dim)
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def load_model(self, model_name, trainable):
        # 使用BERT模型时的特殊处理
        if model_name == 'bert-base-uncased':
            model = BertModel.from_pretrained(model_name, local_files_only=True)
            if trainable:
                peft_config = LoraConfig(
                    target_modules=["query", "value"],
                    inference_mode=False,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                self.set_trainable(model, trainable)
            return model
        else:
            # 处理其他模型
            model = AutoModel.from_pretrained(model_name, local_files_only=True)
            self.set_trainable(model, trainable)
            return model

    @staticmethod
    def set_trainable(model, trainable):
        # 设置模型的可训练性
        for p in model.parameters():
            p.requires_grad = trainable

        # for p in self.model.parameters():
        #     p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        # 每max_length截断
        num = input_ids.shape[1] // self.max_length
        last_hidden_state_list = []
        for i in range(num):
            start_idx = i * self.max_length
            end_idx = (i + 1) * self.max_length
            output = self.model(input_ids=input_ids[:, start_idx:end_idx],
                                attention_mask=attention_mask[:, start_idx:end_idx])
            last_hidden_state = output.last_hidden_state
            # last_token_pool = self.last_token_pool(last_hidden_state, attention_mask[:, start_idx:end_idx])
            #用平均值
            # last_hidden_state_list.append(self.linear(last_token_pool))

            last_hidden_state_list.append(self.linear(last_hidden_state[:, self.target_token_idx, :]))
        return last_hidden_state_list
        # output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = output.last_hidden_state
        # return last_hidden_state[:, self.target_token_idx, :]

    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def SFR_forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.last_token_pool(last_hidden_states=output.last_hidden_state, attention_mask=attention_mask)
        return embeddings

    def mlm_forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state


class RecEncoder_WideDeep(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, total_feature_num,
                 dnn_hidden_units=(300, 300, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary',
                 device='cpu', gpus=None):
        super(RecEncoder_WideDeep, self).__init__(linear_feature_columns, dnn_feature_columns, total_feature_num,
                                                  l2_reg_linear=l2_reg_linear,
                                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed,
                                                  task=task,
                                                  device=device, gpus=gpus)

        # self.field_size = len(self.embedding_dict)
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

            # self.add_regularization_weight(
            #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, text_embedding_list, dense_inputs):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(dense_inputs,
                                                                                  self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        sparse_embedding_list = sparse_embedding_list[:dense_inputs.shape[1]]

        logit = self.linear_model(text_embedding_list, dense_inputs)

        if self.use_dnn:
            dnn_sparse_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_text_input = torch.cat(text_embedding_list, dim=1)
            dnn_input = torch.cat([dnn_text_input, dnn_sparse_input], dim=1)
            # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


class RecEncoder_DeepFM(BaseModel):
    def __init__(self, sparse_feature_columns, sparse_all_feature_columns, total_feature_num, use_fm=True,
                 dnn_hidden_units=(300, 300, 128),
                 l2_reg_linear=0.00000, l2_reg_embedding=0.00000, l2_reg_dnn=0, init_std=0.0001,
                 seed=2024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary',
                 device='cpu', gpus=None):
        super(RecEncoder_DeepFM, self).__init__(sparse_feature_columns, sparse_all_feature_columns, total_feature_num,
                                                l2_reg_linear=l2_reg_linear,
                                                l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed,
                                                task=task,
                                                device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(sparse_all_feature_columns) > 0 and len(dnn_hidden_units) > 0
        # self.embedding_dict=None
        # self.use_dnn = 0
        if use_fm:
            self.fm = FM()
        self.linear_model = Linear(sparse_feature_columns, self.feature_index, total_feature_num, device=device)
        # self.linear_model.embedding_dict=None
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(sparse_all_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)
        # self.out=None

    def forward(self, text_embedding_list, dense_inputs):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            dense_inputs, self.dnn_feature_columns, self.embedding_dict
        )
        sparse_embedding_list = sparse_embedding_list[:dense_inputs.shape[1]]
        logit = self.linear_model(text_embedding_list=text_embedding_list, X=dense_inputs)

        if self.use_fm and len(text_embedding_list) > 0:
            stacked_text_embeddings = torch.stack(text_embedding_list).permute(1, 0, 2)
            fm_input = torch.cat(
                sparse_embedding_list + [stacked_text_embeddings],
                dim=1
            )
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_sparse_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_text_input = torch.cat(text_embedding_list, dim=1)
            dnn_input = torch.cat([dnn_text_input, dnn_sparse_input], dim=1)

            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
            y_pred = self.out(logit)
            return y_pred
            # res = torch.cat([logit, dnn_output], dim=1)

            # return res


class RecEncoder_DCNv2(BaseModel):
    def __init__(self, sparse_feature_columns,
                 sparse_all_feature_columns, total_feature_num, cross_num=2,
                 dnn_hidden_units=(300, 300, 128), l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_cross=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=2024,
                 dnn_dropout=0, low_rank=32, num_experts=4,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(RecEncoder_DCNv2, self).__init__(linear_feature_columns=sparse_feature_columns,
                                               dnn_feature_columns=sparse_all_feature_columns,
                                               total_feature_num=total_feature_num, l2_reg_embedding=l2_reg_embedding,
                                               init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.out = None
        self.linear_model = None
        # self.embedding_dict=None
        # self.projection=nn.Linear(32*6,32)
        # self.dnn_hidden_units = dnn_hidden_units
        # self.cross_num = cross_num
        # self.dnn = DNN(self.compute_input_dim(sparse_all_feature_columns), dnn_hidden_units,
        #                activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
        #                init_std=init_std, device=device)
        # if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
        #     dnn_linear_in_feature = self.compute_input_dim(sparse_all_feature_columns) + dnn_hidden_units[-1]
        # elif len(self.dnn_hidden_units) > 0:
        #     dnn_linear_in_feature = dnn_hidden_units[-1]
        # elif self.cross_num > 0:
        #     dnn_linear_in_feature = self.compute_input_dim(sparse_all_feature_columns)
        #
        # self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1).to(
        #     device)
        # self.crossnet = CrossNetMix(in_features=self.compute_input_dim(sparse_all_feature_columns),
        #                             low_rank=low_rank, num_experts=num_experts,
        #                             layer_num=cross_num, device=device)

        self.to(device)
    def get_embedding(self,text_embedding_list,dense_inputs):
        sparse_embedding_list, _ = self.input_from_feature_columns(
            dense_inputs, self.dnn_feature_columns, self.embedding_dict
        )

        # Remove the first two ID-type features
        sparse_embedding_list = sparse_embedding_list[2:dense_inputs.shape[1]]
        sparse_embedding_list = [embedding.squeeze(1) for embedding in sparse_embedding_list]

        text_embedding = torch.cat(text_embedding_list, dim=1)
        sparse_embedding = torch.cat(sparse_embedding_list, dim=1)

        # text_embedding=self.projection(text_embedding)
        # sparse_embedding=self.projection(sparse_embedding)

        return text_embedding,sparse_embedding


    def modal_fine_forward(self, text_embedding_list, dense_inputs):
        sparse_embedding_list, _ = self.input_from_feature_columns(
            dense_inputs, self.dnn_feature_columns, self.embedding_dict
        )

        # Remove the first two ID-type features
        sparse_embedding_list = sparse_embedding_list[2:dense_inputs.shape[1]]
        sparse_embedding_list = [embedding.squeeze(1) for embedding in sparse_embedding_list]

        batch_size = dense_inputs.shape[0]
        num_features = len(sparse_embedding_list)

        text_embeddings = torch.stack(text_embedding_list)  # [num_features, batch_size, embedding_dim]
        sparse_embeddings = torch.stack(sparse_embedding_list)  # [num_features, batch_size, embedding_dim]

        # 计算所有对的相似度
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=2)
        sparse_embeddings_norm = F.normalize(sparse_embeddings, p=2, dim=2)

        # 计算正样本对的相似度 [num_features, batch_size]
        positive_similarities = torch.sum(text_embeddings_norm * sparse_embeddings_norm, dim=2)

        # 计算负样本对的相似度 [num_features, batch_size, batch_size]
        negative_similarities = torch.bmm(text_embeddings_norm, sparse_embeddings_norm.transpose(1, 2))

        # 创建mask来排除对角线上的正样本
        mask = torch.eye(batch_size, device=dense_inputs.device).bool().unsqueeze(0).expand(num_features, -1, -1)
        negative_similarities.masked_fill_(mask, float('-inf'))

        # 计算对比损失
        temperature = 0.3  # 温度参数，可以调整
        logits = torch.cat([positive_similarities.unsqueeze(2), negative_similarities], dim=2)
        labels = torch.zeros(num_features, batch_size, dtype=torch.long, device=dense_inputs.device)

        contrastive_loss = F.cross_entropy(logits.view(-1, batch_size + 1) / temperature, labels.view(-1))

        return contrastive_loss
        # 细粒度对齐，同一特征的文本和类别特征embedding作为正样本，同一特征不同样本作为负样本

    def modal_forward(self, text_embedding_list, dense_inputs):
        sparse_embedding_list, _ = self.input_from_feature_columns(
            dense_inputs, self.dnn_feature_columns, self.embedding_dict
        )

        # Remove the first two ID-type features
        sparse_embedding_list = sparse_embedding_list[2:dense_inputs.shape[1]]
        sparse_embedding_list = [embedding.squeeze(1) for embedding in sparse_embedding_list]

        text_embedding = torch.cat(text_embedding_list, dim=1)
        sparse_embedding = torch.cat(sparse_embedding_list, dim=1)

        # Calculate cosine similarity efficiently
        text_norm = F.normalize(text_embedding, p=2, dim=1)
        sparse_norm = F.normalize(sparse_embedding, p=2, dim=1)
        similarity_matrix = torch.mm(text_norm, sparse_norm.t())

        # Get positive and negative samples
        positive_similarity = similarity_matrix.diag()
        negative_mask = ~torch.eye(similarity_matrix.size(0), dtype=bool, device=similarity_matrix.device)
        negative_similarity = similarity_matrix[negative_mask].view(similarity_matrix.size(0), -1)

        # Calculate contrastive loss
        loss = self.contrastive_loss(positive_similarity, negative_similarity)
        return loss

    def contrastive_loss(self, positive_similarity, negative_similarity, temperature=0.3):
        """
        计算InfoNCE对比学习损失
        :param positive_similarity: 正样本相似度
        :param negative_similarity: 负样本相似度
        :param temperature: 温度参数，用于控制分布的平滑程度
        :return: 对比学习损失
        """
        positive_similarity = torch.exp(positive_similarity / temperature)
        negative_similarity = torch.sum(torch.exp(negative_similarity / temperature), dim=1)

        loss = -torch.log(positive_similarity / (positive_similarity + negative_similarity))
        return loss.mean()

    def forward(self, text_embedding_list, dense_inputs):
        logit = self.linear_model(text_embedding_list, dense_inputs)

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            dense_inputs, self.dnn_feature_columns, self.embedding_dict
        )

        sparse_embeddings = sparse_embedding_list[:dense_inputs.shape[1]]
        dense_input = combined_dnn_input(sparse_embeddings, dense_value_list)

        text_input = torch.cat(text_embedding_list, dim=1)
        dnn_input = torch.cat([text_input, dense_input], dim=1)

        if self.dnn_hidden_units and self.cross_num:
            deep_output = self.dnn(dnn_input)
            cross_output = self.crossnet(dnn_input)
            stacked_output = torch.cat((cross_output, deep_output), dim=-1)
            # logit = torch.zeros_like(stacked_output)
            logit += self.dnn_linear(stacked_output)  # 没梯度
            # logit = self.dnn_linear(stacked_output)  # 没梯度
            logit = self.out(logit)
            return logit


class RecEncoder_PNN(BaseModel):
    def __init__(self, sparse_feature_columns, sparse_all_feature_columns, total_feature_num,
                 dnn_hidden_units=(300, 300, 128),
                 l2_reg_embedding=0, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False,
                 kernel_type='mat', task='binary', device='cpu', gpus=None):

        super(RecEncoder_PNN, self).__init__(sparse_feature_columns, sparse_all_feature_columns, total_feature_num,
                                             l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                             init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")

        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        self.task = task

        product_out_dim = 0
        num_inputs = self.compute_input_dim(sparse_all_feature_columns, include_dense=False, feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)

        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        self.dnn = DNN(product_out_dim + self.compute_input_dim(sparse_all_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)

        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)
        # self.dnn_linear = None
        self.linear_model = None
        # self.add_regularization_weight(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, text_embedding_list, dense_inputs):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            dense_inputs, self.dnn_feature_columns, self.embedding_dict
        )

        sparse_embeddings = sparse_embedding_list[:dense_inputs.shape[1]]

        linear_sparse_signal = torch.flatten(concat_fun(sparse_embeddings), start_dim=1)
        linear_text_signal = torch.cat(text_embedding_list, dim=1)
        linear_signal = torch.cat([linear_sparse_signal, linear_text_signal], dim=1)

        if self.use_inner:
            text_embeddings = [embedding.unsqueeze(1) for embedding in text_embedding_list]
            inner_product_input = sparse_embeddings + text_embeddings
            inner_product = torch.flatten(self.innerproduct(inner_product_input), start_dim=1)
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        dnn_input = combined_dnn_input([product_layer], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        logit = self.dnn_linear(dnn_output)
        logit = self.out(logit)

        return logit


class RecEncoder_AutoInt(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, total_feature_num, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(300, 300, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu', gpus=None):

        super(RecEncoder_AutoInt, self).__init__(linear_feature_columns, dnn_feature_columns, total_feature_num,
                                                 l2_reg_linear=0,
                                                 l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed,
                                                 task=task,
                                                 device=device, gpus=gpus)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        # field num cant use this
        # field_num = len(self.embedding_dict)
        field_num = len(dnn_feature_columns)
        embedding_size = self.embedding_size

        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])

        self.to(device)

    def forward(self, text_embedding_list, dense_inputs):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(dense_inputs,
                                                                                  self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(text_embedding_list=text_embedding_list, X=dense_inputs)
        sparse_embedding_list = sparse_embedding_list[:dense_inputs.shape[1]]
        sparse_embedding = concat_fun(sparse_embedding_list, axis=1)
        text_embedding = torch.stack(text_embedding_list).permute(1, 0, 2)
        att_input = torch.cat([sparse_embedding, text_embedding], dim=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        dense_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        text_input = torch.cat(text_embedding_list, dim=1)
        dnn_input = torch.cat([text_input, dense_input], dim=1)
        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:  # Error
            pass

        y_pred = self.out(logit)

        return y_pred


class RecEncoder_xDeepFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, total_feature_num, dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(RecEncoder_xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, total_feature_num,
                                                 l2_reg_linear=l2_reg_linear,
                                                 l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed,
                                                 task=task,
                                                 device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            # field_num = len(self.embedding_dict)
            field_num = len(dnn_feature_columns)
            if cin_split_half == True:
                self.featuremap_num = sum(
                    cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)

        self.to(device)

    def forward(self, text_embedding_list, dense_inputs):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(dense_inputs,
                                                                                  self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        sparse_embedding_list = sparse_embedding_list[:dense_inputs.shape[1]]
        linear_logit = self.linear_model(text_embedding_list, dense_inputs)
        text_input = torch.cat(text_embedding_list, dim=1)

        if self.use_cin:
            text_embedding = torch.stack(text_embedding_list).permute(1, 0, 2)
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_input = torch.cat([text_embedding, cin_input], dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        if self.use_dnn:
            dense_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_input = torch.cat([text_input, dense_input], dim=1)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:  # only linear
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:  # linear + CIN
            final_logit = linear_logit + cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:  # linear +　Deep
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:  # linear + CIN + Deep
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        return y_pred


class RecEncoder(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, total_feature_num, dnn_hidden_units=(300, 300, 128),
                 l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(RecEncoder, self).__init__(linear_feature_columns, dnn_feature_columns, total_feature_num,
                                         l2_reg_linear=l2_reg_linear,
                                         l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed,
                                         task=task,
                                         device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       use_bn=dnn_use_bn, init_std=init_std, device=device)
        dnn_linear_in_feature = dnn_hidden_units[-1]
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.linear_model = None

    def forward(self, text_embedding_list, dense_inputs):
        # 直接输出
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(dense_inputs,
                                                                                  self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        sparse_embedding_list = sparse_embedding_list[:dense_inputs.shape[1]]
        # print("sparse_embedding_list, dense_value_list:",sparse_embedding_list, dense_value_list)
        dense_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        text_input = torch.cat(text_embedding_list, dim=1)
        dnn_input = torch.cat([text_input, dense_input], dim=1)
        deep_out = self.dnn(dnn_input)
        logit = self.dnn_linear(deep_out)
        y_pred = self.out(logit)
        return y_pred


class RecEncoder_DCN(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, total_feature_num, cross_num=2,
                 cross_parameterization='vector',
                 dnn_hidden_units=(300, 300, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                 l2_reg_cross=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task='binary', device='cpu', gpus=None):
        super(RecEncoder_DCN, self).__init__(linear_feature_columns, dnn_feature_columns, total_feature_num,
                                             l2_reg_linear=l2_reg_linear,
                                             l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                             device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        # Deep
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       use_bn=dnn_use_bn, init_std=init_std, device=device)

        # Linear
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns) + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            dnn_linear_in_feature = self.compute_input_dim(dnn_feature_columns)

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)

        # Cross
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns),
                                 layer_num=cross_num, parameterization=cross_parameterization, device=device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)
        self.to(device)

    def forward(self, text_embedding_list, dense_inputs):
        logit = self.linear_model(text_embedding_list, dense_inputs)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(dense_inputs,
                                                                                  self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        sparse_embedding_list = sparse_embedding_list[:dense_inputs.shape[1]]
        # print("sparse_embedding_list, dense_value_list:",sparse_embedding_list, dense_value_list)
        dense_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        text_input = torch.cat(text_embedding_list, dim=1)
        dnn_input = torch.cat([text_input, dense_input], dim=1)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((deep_out, cross_out), dim=-1)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:
            pass
        y_pred = self.out(logit)
        return y_pred


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim,
            dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Prediction_Layer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            output_dim,
            hidden_dim,
            dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.activation(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class mlm_Prediction_Layer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            output_dim,
            hidden_dim,
            dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.activation(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x
