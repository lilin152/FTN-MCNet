"""BiomedCLIP dual encoder and LoRA adapters.

L2-normalized vision and report-text embeddings with LoRA adapters.
"""
import torch
from torch import nn
MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

class FusionBackbone(nn.Module):
    """
    BiomedCLIP 双塔 + 图文拼接。
      forward_visual(x) -> [B, 512] L2 归一化图像特征（视觉塔带 LoRA，可训练）
      forward_text(tokens) -> [B, 512] L2 归一化文本特征（仅 freeze_text=False 可训练）
      encode_text_frozen(texts) -> [N, 512] 冻结文本特征
    freeze_text=True  : 文本塔完全冻结，报告文本特征在训练前离线算好缓存
    freeze_text=False : 文本塔加 LoRA（target: query/key/value/dense），端到端训练
    """
    def __init__(self, freeze_text: bool = True,
                 lora_r: int = 16, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 device: str = "cuda", model_id: str = MODEL_ID,
                 with_vision: bool = True):
        super().__init__()
        import open_clip
        from peft import LoraConfig, get_peft_model
        self._open_clip = open_clip
        self.freeze_text = freeze_text
        self.model_id = model_id
        self.device = device
        self.feat_dim = 512
        self.with_vision = with_vision

        hf_id = f"hf-hub:{model_id}"
        self.model, _, _ = open_clip.create_model_and_transforms(hf_id, pretrained=hf_id)

        # ---- 视觉塔 LoRA（with_vision=False 时冻结）----
        # 关键: CustomTextCLIP.encode_image 内部调用 self.visual(...)，
        #       必须直接替换 self.model.visual，LoRA 才会真正生效
        if with_vision:
            vis_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 target_modules=["qkv", "proj", "fc1", "fc2"], bias="none")
            self.model.visual = get_peft_model(self.model.visual, vis_cfg)
        else:
            for p in self.model.visual.parameters():
                p.requires_grad = False

        # ---- 文本塔 ----
        # CustomTextCLIP.encode_text 内部调用 self.text(...)，直接替换 self.model.text
        if freeze_text:
            for p in self.model.text.parameters():
                p.requires_grad = False
        else:
            txt_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 target_modules=["query", "key", "value", "dense"], bias="none")
            self.model.text = get_peft_model(self.model.text, txt_cfg)

        # The classification heads do not use CLIP's contrastive temperature.
        # Freeze it explicitly so the trainable parameters are limited to LoRA
        # adapters and the task-specific heads described in the manuscript.
        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad = False

        self.model = self.model.to(device)

        # CLIP 标准归一化参数（与 BiomedCLIPBackbone 一致）
        self.register_buffer('_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        # 训练/推理可统计
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"[FusionBackbone] freeze_text={freeze_text} | "
              f"total={total:,} trainable={trainable:,} ({trainable/total*100:.2f}%)")
        for n in trainable_names[:6]:
            print(f"    trainable: {n}")
        if len(trainable_names) > 6:
            print(f"    ... and {len(trainable_names)-6} more")
        # 验证: 文本塔参数在 frozen 时必须全部不可训练
        if freeze_text:
            txt_trainable = [n for n, p in self.model.text.named_parameters() if p.requires_grad]
            assert len(txt_trainable) == 0, f"frozen mode has trainable text params: {txt_trainable}"

    def forward_visual(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B,512] L2 归一化图像特征（与 BiomedCLIPBackbone.forward 一致）"""
        x = (x.to(self.device) - self._mean.to(self.device)) / self._std.to(self.device)
        return self.model.encode_image(x, normalize=True)

    def forward_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B,L] tokens -> [B,512] L2 归一化文本特征（freeze_text=False 时走这里，可训练）"""
        return self.model.encode_text(tokens, normalize=True)

    @torch.no_grad()
    def encode_text_frozen(self, texts) -> torch.Tensor:
        """冻结文本特征：[N] 文本 -> [N,512] L2 归一化（离线缓存用）"""
        tokenizer = self._open_clip.get_tokenizer(f"hf-hub:{self.model_id}")
        tokens = tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens, normalize=True)

    @torch.no_grad()
    def get_tokens(self, texts) -> torch.Tensor:
        """文本 -> tokens [N,L]（freeze_text=False 时预计算，CPU）"""
        tokenizer = self._open_clip.get_tokenizer(f"hf-hub:{self.model_id}")
        return tokenizer(texts).cpu()

    @property
    def logit_scale(self) -> torch.Tensor:
        """可学习温度系数（exp 后）"""
        return self.model.logit_scale.exp()


# ============================================================
#  公共：图文特征提取
# ============================================================
def _feats(backbone: FusionBackbone, x: torch.Tensor, txt: torch.Tensor):
    """
    返回 (img_feat [B,512], txt_feat [B,512])，均 L2 归一化。
      freeze_text=True  : txt 为离线缓存文本特征 [B,512]
      freeze_text=False : txt 为 tokens [B,L]，在线过文本塔（有梯度）
    """
    img_feat = backbone.forward_visual(x)
    if backbone.freeze_text:
        txt_feat = txt.to(img_feat.device)
    else:
        txt_feat = backbone.forward_text(txt.to(backbone.device))
    return img_feat, txt_feat


