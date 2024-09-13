from typing import Dict, Type

import torch.nn as nn

from layers.modules_mix import *


class bertCTRModel(nn.Module):
    def __init__(
            self,
            cfg,
            text_encoder_model,
            sparse_feature_columns=None,
            sparse_all_feature_columns=None,
            struct_feature_num=None,):
        super(bertCTRModel, self).__init__()
        self.text_encoder = TextEncoder(text_encoder_model, cfg)
        backbone_map: Dict[str, Type[nn.Module]] = {
            "DeepFM": RecEncoder_mix_DeepFM,
            "DCNv2": RecEncoder_mix_DCNv2,
            "PNN": RecEncoder_mix_PNN,
            "AutoInt": RecEncoder_mix_AutoInt,
            "xDeepFM": RecEncoder_mix_xDeepFM,
            "DCN": RecEncoder_mix_DCN,
            "Recoder": RecEncoder,
            "widedeep": RecEncoder_mix_WideDeep
        }

        if cfg.backbone in backbone_map:
            self.rec_encoder = backbone_map[cfg.backbone](cfg,
                sparse_feature_columns, sparse_all_feature_columns, struct_feature_num
            )
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")
        if cfg.llm == "SFR":
            self.text_encoder.forward = self.text_encoder.SFR_forward

    #
    def forward(self, batch):
        text_features_list1 = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        rec_pred, text_pred, y_pred = self.rec_encoder(text_features_list1, batch['rec_data'])

        return rec_pred, text_pred, y_pred
