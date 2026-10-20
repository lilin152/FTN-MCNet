"""BiomedCLIP feature concatenation and independent binary/four-class MLP heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FusionBackbone, _feats


def build_classification_head(in_dim: int, h1: int = 256, h2: int = 128,
                                  dropout: float = 0.3, out_dim: int = 2) -> nn.Sequential:
    """三层独立 FC 分类头：Linear(h1) + ReLU + Dropout → Linear(h2) + ReLU + Dropout → Linear(out)"""
    return nn.Sequential(
        nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, out_dim),
    )


class MultimodalFusionClassifier(nn.Module):
    """
    图文直接拼接 + binary/four_class 两个独立三层 FC 头。
    forward(x, txt) -> (binary_logits[B,2], four_class_logits[B,4])；预测用 four_class softmax。
    """

    def __init__(self, backbone: FusionBackbone, feat_dim: int = 1024,
                 hidden1: int = 256, hidden2: int = 128, dropout: float = 0.3):
        super().__init__()
        assert isinstance(backbone, FusionBackbone)
        self.backbone = backbone
        self.feat_dim = feat_dim
        # binary/four_class 分类头：参数完全独立，不使用类别 prompt / logit_scale
        self.binary_head = build_classification_head(feat_dim, hidden1, hidden2, dropout, out_dim=2)
        self.four_class_head = build_classification_head(feat_dim, hidden1, hidden2, dropout, out_dim=4)

    def forward(self, x: torch.Tensor, txt: torch.Tensor):
        """x:[B,3,224,224]；txt: unfrozen tokens[B,256]（或 frozen 特征[B,512]）"""
        img_feat, txt_feat = _feats(self.backbone, x, txt)   # 均 [B,512] L2
        z = torch.cat([img_feat, txt_feat], dim=1)           # [B,1024]
        return self.binary_head(z), self.four_class_head(z)

    @torch.no_grad()
    def predict_four_class(self, x: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        """推理：four_class logits 的 softmax 作为 4 类概率"""
        _, four_class_logits = self.forward(x, txt)
        return F.softmax(four_class_logits, dim=1)


def build_model(device: str = "cuda"):
    """Build the multimodal classifier with visual and text LoRA adapters."""
    backbone = FusionBackbone(freeze_text=False, device=device)
    return MultimodalFusionClassifier(backbone).to(device)


