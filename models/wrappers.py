"""
Composite model wrappers used for GradCAM visualisation and standalone inference.
Author: Ankit Yadav
"""
import warnings
import torch.nn as nn


class SIGLIPWithMLP(nn.Module):
    """Wraps a vision backbone (SigLIP / DINOv2 / ResNet) together with an MLP
    head so that GradCAM can be applied end-to-end.

    Args:
        base_model: the vision encoder (e.g. SigLIP2 ``AutoModel``).
        mlp_head: the quality-prediction MLP (e.g. ``MLP3_Gated``).
        device: torch device.
        layer: layer index used by perception-style encoders.
        resnet: set True when the backbone is a ResNet.
    """

    def __init__(self, base_model, mlp_head, device, layer=18, resnet=False):
        super().__init__()
        self.siglip   = base_model
        self.mlp_head = mlp_head
        self.device   = device
        self.layer    = layer
        self.resnet   = resnet

    def forward(self, inputs):
        if self.resnet:
            warnings.warn(
                "ResNet152 backbone detected — using pooler_output. "
                "If this is not intended, check the backbone type."
            )
            features = self.siglip(**inputs).pooler_output
            features = features.squeeze(-1).squeeze(-1)
        else:
            try:
                features = self.siglip.get_image_features(inputs)
            except Exception:
                warnings.warn(
                    "SigLIP get_image_features failed — falling back to "
                    "DINOv2-style average-pooled last_hidden_state."
                )
                try:
                    features = self.siglip(inputs).last_hidden_state.mean(dim=1)
                except Exception:
                    warnings.warn(
                        "DINOv2 fallback failed — falling back to "
                        "Perception-style encode_image_layers. "
                        "If this is not intended, check the backbone type."
                    )
                    features = self.siglip.encode_image_layers(
                        inputs, layer_idx=self.layer
                    )

        scores = self.mlp_head(features)  # (B, 1)
        return scores.squeeze(1)          # (B,)
