# -*- coding: utf-8 -*-
"""
Detectron2 backbone wrapper for DINO(v3/v2)-style ViTs with robust HuggingFace loading.

Priority:
1) LOCAL REPO  (torch.hub local)
2) HUGGING FACE (prefer huggingface_hub with token; fallback to urllib with Authorization)
3) TIMM fallback
4) REMOTE HUB (last resort)

Env / Config:
- MODEL.DINOV3.NAME            (e.g. dinov3_vit7b16)
- MODEL.DINOV3.PRETRAINED      (bool)  -- set False when using HF weights
- MODEL.DINOV3.OUT_FEATURES    (default ["res5"])
- MODEL.DINOV3.LOCAL_REPO      / DINO_LOCAL_REPO
- MODEL.DINOV3.HF_REPO         / DINO_HF_REPO      (default: facebook/dinov3-vit7b16-pretrain-lvd1689m)
- MODEL.DINOV3.HF_FILE         / DINO_HF_FILE      (optional, exact filename)
- MODEL.DINOV3.HF_LOCAL        / DINO_HF_LOCAL     (optional, direct local path to weight)
- MODEL.DINOV3.ARCH            / DINO_ARCH         (timm arch)
- MODEL.DINOV3.CKPT            / DINO_CKPT         (local ckpt for timm)
- Token (optional): DINO_HF_TOKEN / HUGGINGFACEHUB_API_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN
- Mirror (optional): HF_ENDPOINT (e.g. https://hf-mirror.com)
"""

import os
from typing import Optional, List

import torch
from torch import nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

# optional timm
try:
    from timm import create_model
except Exception:
    create_model = None  # optional

# optional huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # optional


# ---------------------- utils ----------------------

def _exists_file(p: Optional[str]) -> bool:
    return bool(p) and os.path.isfile(p)  # type: ignore

def _exists_dir(p: Optional[str]) -> bool:
    return bool(p) and os.path.isdir(p)  # type: ignore

def _mkdir_p(path: str):
    os.makedirs(path, exist_ok=True)

def _torchhub_ckpt_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    _mkdir_p(d)
    return d

def _hf_cache_dir() -> str:
    # Respect HF cache if set; else default to ~/.cache/huggingface
    d = os.environ.get("HUGGINGFACE_HUB_CACHE",
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    _mkdir_p(d)
    return d

def _get_hf_token() -> Optional[str]:
    for k in ("DINO_HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    return None


# ---------------------- builders ----------------------

def _try_load_from_local_repo(local_repo: str, model_name: str, pretrained: bool):
    print(f"[DINO-backbone] Loading from LOCAL_REPO={local_repo}, name={model_name}, pretrained={pretrained}")
    return torch.hub.load(local_repo, model_name, source="local", pretrained=pretrained)

def _try_load_from_remote_hub(model_name: str, pretrained: bool):
    print(f"[DINO-backbone] Loading from REMOTE HUB name={model_name}, pretrained={pretrained}")
    return torch.hub.load("facebookresearch/dinov3", model_name, pretrained=pretrained)

def _timm_make_model(arch: str, pretrained: bool, ckpt_path: Optional[str]):
    if create_model is None:
        raise RuntimeError("timm 未安装，且本地/远端 hub 不可用。请安装 timm 或提供本地 dinov3 repo。")
    print(f"[DINO-backbone] Fallback to timm arch={arch}, pretrained={pretrained}, ckpt={ckpt_path}")
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool="")
    if _exists_file(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state: state = state["model"]
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    elif ckpt_path:
        raise FileNotFoundError(f"指定的 CKPT 不存在: {ckpt_path}")
    return model


# ---------------------- HuggingFace helpers ----------------------

def _hf_candidates(specified: Optional[str]) -> List[str]:
    if specified:
        return [specified]
    return [
        "dinov3_vit7b16_pretrain_lvd1689m.pth",
        "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
        "pytorch_model.bin",
        "model.pth",
        "checkpoint.pth",
        "weights.pth",
        "model.safetensors",
    ]

def _find_local_any(candidates: List[str]) -> Optional[str]:
    # 1) explicit torch hub checkpoints dir
    for base in (_torchhub_ckpt_dir(), _hf_cache_dir()):
        for fname in candidates:
            p = os.path.join(base, fname)
            if _exists_file(p):
                print(f"[DINO-backbone][HF] Found local cached: {p}")
                return p
    return None

def _download_from_hf(repo: str, file_candidates: List[str], prefer_cache_dir: str, token: Optional[str]) -> str:
    """
    Prefer huggingface_hub if available (handles 401, proxies, mirrors),
    otherwise fallback to urllib with Authorization header when token provided.
    """
    last_err = None

    # Try huggingface_hub first
    if hf_hub_download is not None:
        for fname in file_candidates:
            try:
                print(f"[DINO-backbone][HF] hf_hub_download: {repo} :: {fname}")
                local = hf_hub_download(
                    repo_id=repo,
                    filename=fname,
                    token=token,
                    local_dir=prefer_cache_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                if _exists_file(local):
                    return local
            except Exception as e:
                last_err = e
                print(f"[DINO-backbone][HF] hf_hub_download failed: {e}")

    # Fallback: urllib (bearer token optional)
    import urllib.request, urllib.error, shutil
    opener = urllib.request.build_opener()
    if token:
        opener.addheaders = [("Authorization", f"Bearer {token}")]
    urllib.request.install_opener(opener)

    for fname in file_candidates:
        url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{repo}/resolve/main/{fname}"
        out_path = os.path.join(prefer_cache_dir, fname)
        try:
            print(f"[DINO-backbone][HF] URL open: {url}")
            with urllib.request.urlopen(url, timeout=60) as resp, open(out_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            if os.path.getsize(out_path) < (1 << 20):
                raise RuntimeError(f"下载文件过小：{out_path}")
            return out_path
        except Exception as e:
            last_err = e
            try:
                if os.path.exists(out_path): os.remove(out_path)
            except Exception:
                pass
            print(f"[DINO-backbone][HF] urllib failed: {e}")

    raise RuntimeError(f"无法从 Hugging Face 获取权重（repo={repo}）。最后错误: {last_err}")

def _load_ckpt_any(path: str) -> dict:
    # support .safetensors
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state = load_file(path)
            return state
        except Exception as e:
            raise RuntimeError(f"加载 safetensors 失败: {e}")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    return state


# ---------------------- tokens & shapes ----------------------

def _infer_patch_size(model: nn.Module) -> int:
    patch = None
    if hasattr(model, "patch_embed"):
        ps = getattr(model.patch_embed, "patch_size", None)
        if isinstance(ps, (tuple, list)): patch = int(ps[0])
        elif isinstance(ps, int): patch = ps
    return int(patch if patch else 16)

def _infer_embed_dim(model: nn.Module) -> Optional[int]:
    for k in ["embed_dim", "num_features", "num_features_att", "dim"]:
        if hasattr(model, k) and isinstance(getattr(model, k), int):
            return int(getattr(model, k))
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        return int(model.head.in_features)
    return None

def _extract_tokens_any(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                return out["x_norm_patchtokens"]
            for k in ["x", "tokens", "patch_tokens"]:
                if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() == 3:
                    t = out[k]
                    n = t.size(1); s = int(n ** 0.5)
                    if s * s != n and n > 1: t = t[:, 1:, :]
                    return t
        elif isinstance(out, torch.Tensor) and out.dim() == 3:
            t = out; n = t.size(1); s = int(n ** 0.5)
            if s * s != n and n > 1: t = t[:, 1:, :]
            return t
    gil = getattr(model, "get_intermediate_layers", None)
    if callable(gil):
        toks = gil(x, n=1, return_class_token=False)
        if isinstance(toks, (list, tuple)) and len(toks) > 0 and isinstance(toks[0], torch.Tensor):
            return toks[0]
    raise RuntimeError("无法提取 (B,N,C) patch tokens；请使用 dinov3 兼容权重/架构。")


# ---------------------- D2 backbone ----------------------

@BACKBONE_REGISTRY.register()
class D2Dinov3(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        hub_name: Optional[str] = getattr(cfg.MODEL.DINOV3, "NAME", None)
        pretrained: bool = bool(getattr(cfg.MODEL.DINOV3, "PRETRAINED", True))
        self._out_features = list(getattr(cfg.MODEL.DINOV3, "OUT_FEATURES", ["res5"]))

        local_repo = getattr(cfg.MODEL.DINOV3, "LOCAL_REPO", None) or os.environ.get("DINO_LOCAL_REPO")
        timm_arch = getattr(cfg.MODEL.DINOV3, "ARCH", None) or os.environ.get("DINO_ARCH")
        timm_ckpt = getattr(cfg.MODEL.DINOV3, "CKPT", None) or os.environ.get("DINO_CKPT")

        hf_repo = getattr(cfg.MODEL.DINOV3, "HF_REPO", None) or os.environ.get("DINO_HF_REPO") or \
                  "facebook/dinov3-vit7b16-pretrain-lvd1689m"
        hf_file = getattr(cfg.MODEL.DINOV3, "HF_FILE", None) or os.environ.get("DINO_HF_FILE")
        hf_local = getattr(cfg.MODEL.DINOV3, "HF_LOCAL", None) or os.environ.get("DINO_HF_LOCAL")
        hf_token = _get_hf_token()

        self.model = None
        last_error = None

        # 0) LOCAL REPO
        if _exists_dir(local_repo) and hub_name:
            try:
                self.model = _try_load_from_local_repo(local_repo, hub_name, pretrained=pretrained)
            except Exception as e:
                last_error = e
                print(f"[DINO-backbone] Local repo failed: {e}")

        # 1) HUGGINGFACE: build code via hub (pretrained=False), then load HF weight
        if self.model is None and hub_name and hf_repo:
            try:
                base = _try_load_from_remote_hub(hub_name, pretrained=False)  # only code/arch (uses cache)
                candidates = _hf_candidates(hf_file)

                # (a) explicit local path
                if _exists_file(hf_local):
                    ckpt_path = hf_local
                    print(f"[DINO-backbone][HF] Using HF_LOCAL: {ckpt_path}")
                else:
                    # (b) already cached locally?
                    local_cached = _find_local_any(candidates)
                    if local_cached:
                        ckpt_path = local_cached
                    else:
                        # (c) download from HF with token/mirror/proxy
                        prefer_dir = _torchhub_ckpt_dir()  # align with torch hub layout
                        ckpt_path = _download_from_hf(hf_repo, candidates, prefer_dir, token=hf_token)

                state = _load_ckpt_any(ckpt_path)
                missing, unexpected = base.load_state_dict(state, strict=False)
                if missing:   print(f"[DINO-backbone] Missing keys: {len(missing)}")
                if unexpected:print(f"[DINO-backbone] Unexpected keys: {len(unexpected)}")
                self.model = base
            except Exception as e:
                last_error = e
                print(f"[DINO-backbone] HF path failed: {e}")

        # 2) TIMM FALLBACK
        if self.model is None and timm_arch:
            try:
                self.model = _timm_make_model(timm_arch, pretrained=pretrained, ckpt_path=timm_ckpt)
            except Exception as e:
                last_error = e
                print(f"[DINO-backbone] timm fallback failed: {e}")

        # 3) REMOTE HUB (last resort; may hit forbidden hosts if pretrained=True)
        if self.model is None and hub_name:
            try:
                self.model = _try_load_from_remote_hub(hub_name, pretrained=pretrained)
            except Exception as e:
                last_error = e
                print(f"[DINO-backbone] Remote hub failed: {e}")

        if self.model is None:
            raise RuntimeError(
                "无法构建 DINO backbone：\n"
                f"  - LOCAL_REPO={local_repo}\n"
                f"  - HUB NAME={hub_name}\n"
                f"  - HF_REPO={hf_repo}\n"
                f"  - TIMM ARCH={timm_arch}\n"
                f"最后错误: {last_error}"
            )

        # meta
        self._patch_size = _infer_patch_size(self.model)
        embed_dim = _infer_embed_dim(self.model)
        if embed_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                tokens = _extract_tokens_any(self.model, dummy)
                embed_dim = int(tokens.size(-1))
        self._embed_dim = int(embed_dim)
        self._out_feature_channels = {"res5": self._embed_dim}
        self._out_feature_strides = {"res5": self._patch_size}

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4, f"DINO backbone expects (N,C,H,W). Got {tuple(x.shape)}"
        B, C, H, W = x.shape
        p = int(self._patch_size)

        # 约束：输入应按 patch_size 可整除；Detectron2 可通过 SIZE_DIVISIBILITY 自动补齐
        if (H % p) != 0 or (W % p) != 0:
            raise RuntimeError(
                f"Input size {H}x{W} is not divisible by patch_size={p}. "
                f"Please set INPUT.SIZE_DIVISIBILITY: {p} (or pad/resize appropriately)."
            )

        tokens = _extract_tokens_any(self.model, x)  # (B, N[, +1], C)
        gh, gw = H // p, W // p
        n = tokens.size(1)

        # 若包含 cls token（常见 N = gh*gw + 1），则去掉
        if n == gh * gw + 1:
            tokens = tokens[:, 1:, :]
            n = tokens.size(1)

        if n != gh * gw:
            raise RuntimeError(
                f"Token count mismatch: got N={n}, but expected gh*gw={gh * gw} "
                f"for input {H}x{W} with patch_size={p}. "
                f"Check preprocess / interpolation."
            )

        # (B, N, C) -> (B, C, gh, gw)
        feats = tokens.permute(0, 2, 1).reshape(B, self._embed_dim, gh, gw)
        return {"res5": feats}

    def output_shape(self):
        return {
            k: ShapeSpec(
                channels=self._out_feature_channels[k],
                stride=self._out_feature_strides[k],
            )
            for k in self._out_features
        }
