from transformers import AutoModel
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from transformers import XCLIPVisionModel
class XCLIP(nn.Module):
    def __init__(
        self, pretrained_model_path
    ):
        super(XCLIP, self).__init__()
      
        xclip_model, loading_info = AutoModel.from_pretrained(
            pretrained_model_path, output_loading_info=True
        )
        self.backbone = xclip_model.vision_model if hasattr(xclip_model, 'vision_model') else xclip_model
        vision_prefix = 'vision_model.' if hasattr(xclip_model, 'vision_model') else ''
        missing_vision_weights = [
            key for key in loading_info['missing_keys']
            if key.startswith(vision_prefix)
        ]
        if missing_vision_weights:
            raise RuntimeError(
                "The checkpoint did not load X-CLIP vision weights. Missing keys include: "
                + ', '.join(missing_vision_weights[:10])
            )
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.backbone(images, output_hidden_states=True)
        sequence_output = outputs['pooler_output'].reshape(b, t, -1)
        video_level_features = self.fc_norm(sequence_output.mean(1))
        pred = self.head(video_level_features)

        return pred


