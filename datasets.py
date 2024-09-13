import numpy as np
import torch

from preprocessing.inputs import build_input_features

import torch
import numpy as np


class BertCTRDataset(torch.utils.data.Dataset):
    def __init__(self, struct_data, text_data, text_label, sparse_feature_columns,
                 sparse_all_feature_columns, tokenizer, max_length):
        self.max_length = max_length
        self.feature_index = build_input_features(sparse_feature_columns)
        self.rec_data = self.process_struct_data(struct_data)
        self.text_data = text_data.drop(columns=['label'])
        self.text_label = text_label
        self.tokenizer = tokenizer
        self.num_field = self.rec_data.shape[1]

    def process_struct_data(self, struct_data):
        if isinstance(struct_data, dict):
            struct_data = [struct_data[feature] for feature in self.feature_index]
        struct_data = [np.expand_dims(data, axis=1) if len(data.shape) == 1 else data for data in struct_data]
        return torch.from_numpy(np.concatenate(struct_data, axis=-1)).float()

    def encode_text(self, text_each_row):
        input_ids_list, attention_mask_list = [], []
        for text in text_each_row:
            encoded_input = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoded_input['input_ids'])
            attention_mask_list.append(encoded_input['attention_mask'])

        return torch.cat(input_ids_list, dim=1).flatten(), torch.cat(attention_mask_list, dim=1).flatten()

    def __getitem__(self, idx):
        text_each_row = self.text_data.iloc[idx]
        input_ids, attention_mask = self.encode_text(text_each_row)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_length': torch.tensor(self.max_length, dtype=torch.int),
            'rec_data': self.rec_data[idx],
            'label': torch.tensor(self.text_label[idx], dtype=torch.int)
        }

    def __len__(self):
        return len(self.text_data)
