from typing import Dict, Type

import torch.nn as nn

from layers.modules import TextEncoder, RecEncoder_DeepFM, RecEncoder_PNN, RecEncoder_DCNv2, \
    RecEncoder_AutoInt, RecEncoder_xDeepFM, RecEncoder_DCN, RecEncoder, RecEncoder_WideDeep


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
            "DeepFM": RecEncoder_DeepFM,
            "DCNv2": RecEncoder_DCNv2,
            "PNN": RecEncoder_PNN,
            "AutoInt": RecEncoder_AutoInt,
            "xDeepFM": RecEncoder_xDeepFM,
            "DCN": RecEncoder_DCN,
            "Recoder": RecEncoder,
            "widedeep": RecEncoder_WideDeep
        }

        if cfg.backbone in backbone_map:
            self.rec_encoder = backbone_map[cfg.backbone](
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
        logits = self.rec_encoder(text_features_list1, batch['rec_data'])

        return logits
