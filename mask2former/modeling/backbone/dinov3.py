import torch
from torch import nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class D2Dinov3(Backbone):
    """Detectron2 wrapper for DINOv3 vision transformers."""

    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.DINOV3.NAME
        pretrained = cfg.MODEL.DINOV3.PRETRAINED
        self.model = torch.hub.load(
            "facebookresearch/dinov3", model_name, pretrained=pretrained
        )
        patch = self.model.patch_embed.patch_size
        if isinstance(patch, tuple):
            patch = patch[0]
        self._out_features = cfg.MODEL.DINOV3.OUT_FEATURES
        self._out_feature_channels = {"res5": self.model.embed_dim}
        self._out_feature_strides = {"res5": patch}

    def forward(self, x):
        assert x.dim() == 4, f"DINOv3 takes an input of shape (N,C,H,W). Got {x.shape}"
        feats = self.model.forward_features(x)["x_norm_patchtokens"]
        b, n, c = feats.shape
        h = w = int(n ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(b, c, h, w)
        return {"res5": feats}

    def output_shape(self):
        return {
            k: ShapeSpec(
                channels=self._out_feature_channels[k],
                stride=self._out_feature_strides[k],
            )
            for k in self._out_features
        }