#!/usr/bin/env python3
"""Compare visual embedding geometry across NEW vision / vision-language models.

This script uses the same protocol as the previous experiment:
- Load selected images from selected_images_seed42.txt (must exist, no resampling).
- Extract one vision-only embedding per image (pre-LLM).
- Compute cosine similarity statistics over all unordered pairs.

Model IDs are explicitly listed in MODEL_SPECS below.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure deterministic CUBLAS behavior when determinism is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid flash-attn import/use to sidestep incompatible binaries.
os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLED", "1")
os.environ.setdefault("TORCH_SDPA_ENABLE_FLASH", "0")

# Best-effort: force-disable flash-attn checks in older transformers builds.
try:
    from transformers.modeling_utils import PreTrainedModel

    def _no_flash_attn_2(self, *args, **kwargs):
        return False

    PreTrainedModel._flash_attn_2_can_dispatch = _no_flash_attn_2
except Exception:
    pass


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    family: str


# Exact HF model IDs used (documented per user request):
MODEL_SPECS = [
    # Vision-only
    # ModelSpec("DINOv3-H+", "facebook/dinov3-vith16plus-pretrain-lvd1689m", "dinov3"),
    # Vision-language
    # ModelSpec("Qwen3-VL-8B", "Qwen/Qwen3-VL-8B-Instruct", "qwen3_vl"),
    # ModelSpec("Kosmos-2.5", "microsoft/kosmos-2.5", "kosmos2_5"),
    # ModelSpec("Phi-3-Vision", "microsoft/Phi-3-vision-128k-instruct", "phi3_vision"),
    # ModelSpec("Phi-4-Multimodal", "microsoft/Phi-4-multimodal-instruct", "phi4_mm"),
    # SigLIP2 Shape Optimized 400M variants (from https://huggingface.co/blog/siglip2)
    ModelSpec("SigLIP2-So400M-P14-224", "google/siglip2-so400m-patch14-224", "siglip2"),
    ModelSpec("SigLIP2-So400M-P14-384", "google/siglip2-so400m-patch14-384", "siglip2"),
    ModelSpec("SigLIP2-So400M-P16-256", "google/siglip2-so400m-patch16-256", "siglip2"),
    ModelSpec("SigLIP2-So400M-P16-384", "google/siglip2-so400m-patch16-384", "siglip2"),
    ModelSpec("SigLIP2-So400M-P16-512", "google/siglip2-so400m-patch16-512", "siglip2"),
    ModelSpec("SigLIP2-So400M-NaFlex", "google/siglip2-so400m-patch16-naflex", "siglip2"),
]


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some ops may not support deterministic mode; keep going.
        pass


def choose_device(device_arg: str) -> Tuple[str, torch.dtype]:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if device.startswith("cuda") and torch.cuda.is_available():
        # Prefer float16 for broad compatibility across model ops.
        dtype = torch.float16
        try:
            # Force SDPA without flash to avoid flash-attn binary issues.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    else:
        dtype = torch.float32
    return device, dtype


def load_selected_images(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {txt_path}. This script will not resample images."
        )
    paths = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
    if len(paths) != 100:
        raise ValueError(f"Expected 100 image paths in {txt_path}, found {len(paths)}")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image files from {txt_path} (first: {missing[0]})")
    return paths


def prepare_images(image_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return images


def move_to_device(inputs: Dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _get_submodule(model, path: str):
    cur = model
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def get_vision_module(model, family: str):
    # Best-effort mapping to a vision encoder (pre-LLM / pre-fusion).
    if hasattr(model, "get_vision_tower"):
        try:
            vt = model.get_vision_tower()
            if isinstance(vt, list):
                vt = vt[0]
            if hasattr(vt, "vision_tower"):
                vt = vt.vision_tower
            if hasattr(vt, "vision_model"):
                vt = vt.vision_model
            return vt
        except Exception:
            pass

    candidates = [
        "vision_model",
        "vision_tower",
        "vision_encoder",
        "image_encoder",
        "visual",
        "model.vision_model",
        "model.vision_tower",
        "model.vision_encoder",
        "model.image_encoder",
        "model.visual",
        "model.model.vision_model",
        "model.model.vision_tower",
        "model.model.visual",
    ]

    if family in {"qwen3_vl"}:
        candidates = [
            "model.visual",
            "visual",
            "vision_model",
            "vision_tower",
            "model.vision_model",
            "model.vision_tower",
            "model.model.visual",
        ] + candidates
    if family in {"phi3_vision"}:
        candidates = [
            "model.vision_embed_tokens.img_processor.vision_model",
            "vision_embed_tokens.img_processor.vision_model",
            "model.vision_embed_tokens.img_processor",
            "vision_embed_tokens.img_processor",
        ] + candidates
    if family in {"phi4_mm"}:
        candidates = [
            "model.embed_tokens_extend.image_embed.img_processor",
            "embed_tokens_extend.image_embed.img_processor",
            "model.model.embed_tokens_extend.image_embed.img_processor",
        ] + candidates
    if family in {"emu3"}:
        candidates = [
            "model.vqmodel",
            "vqmodel",
            "model.model.vqmodel",
        ] + candidates

    for path in candidates:
        sub = _get_submodule(model, path)
        if sub is not None:
            if hasattr(sub, "vision_tower"):
                sub = sub.vision_tower
            if hasattr(sub, "vision_model"):
                sub = sub.vision_model
            return sub
    return None


def _filter_kwargs_for_module(module, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    try:
        sig = inspect.signature(module.forward)
        params = sig.parameters
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return inputs
        allowed = set(params.keys())
    except Exception:
        return inputs

    # Handle common alt names
    filtered = dict(inputs)
    if "image_grid_thw" in filtered and "grid_thw" not in filtered:
        filtered["grid_thw"] = filtered["image_grid_thw"]
    if "grid_thw" in filtered and "image_grid_thw" not in filtered:
        filtered["image_grid_thw"] = filtered["grid_thw"]
    if "pixel_values" in filtered and "image_pixel_values" not in filtered:
        filtered["image_pixel_values"] = filtered["pixel_values"]
    if "image_attention_mask" in filtered and "patch_attention_mask" not in filtered:
        try:
            filtered["patch_attention_mask"] = filtered["image_attention_mask"].to(torch.bool)
        except Exception:
            filtered["patch_attention_mask"] = filtered["image_attention_mask"]

    return {k: v for k, v in filtered.items() if k in allowed}


def _filter_kwargs_for_callable(func, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return inputs
        allowed = set(params.keys())
    except Exception:
        return inputs

    return {k: v for k, v in inputs.items() if k in allowed}


def _get_last_hidden_state(outputs) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise ValueError("Could not find last_hidden_state in vision outputs")


def _extract_image_embedding(outputs) -> Optional[torch.Tensor]:
    if outputs is None:
        return None
    if torch.is_tensor(outputs):
        return outputs
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "image_model_output") and outputs.image_model_output is not None:
        return _extract_image_embedding(outputs.image_model_output)
    if hasattr(outputs, "vision_model_output") and outputs.vision_model_output is not None:
        return _extract_image_embedding(outputs.vision_model_output)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return _pool_tokens(outputs.last_hidden_state)
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return _extract_image_embedding(outputs[0])
    return None


def _pool_tokens(hidden: torch.Tensor) -> torch.Tensor:
    # hidden: [B, T, D] or [B, D]
    if hidden.dim() == 2:
        pooled = hidden
    else:
        if hidden.size(1) > 1:
            pooled = hidden[:, 1:, :].mean(dim=1)  # exclude CLS if present
        else:
            pooled = hidden[:, 0, :]
    return F.normalize(pooled, dim=-1)


def _warn(msg: str) -> None:
    print(f"Warning: {msg}")


def _is_cuda_device(dev) -> bool:
    if isinstance(dev, str):
        return dev.startswith("cuda")
    if isinstance(dev, torch.device):
        return dev.type == "cuda"
    return False


def load_model_and_processor(spec: ModelSpec, device: str, dtype: torch.dtype):
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor

    if spec.family == "dinov3":
        processor = AutoImageProcessor.from_pretrained(spec.hf_id)
        # DINOv3 is numerically sensitive; use float32 to avoid NaNs.
        model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=torch.float32)
        model.eval().to(device)
        return model, processor

    # Patch missing cache helper for some multimodal models on older transformers builds.
    try:
        from transformers import cache_utils

        if not hasattr(cache_utils, "SlidingWindowCache"):
            class SlidingWindowCache(cache_utils.DynamicCache):
                pass

            cache_utils.SlidingWindowCache = SlidingWindowCache
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)

    model = None
    common_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if spec.family == "phi3_vision":
        # Phi-3 Vision: use eager attention (SDPA not supported).
        common_kwargs["attn_implementation"] = "eager"
    elif spec.family == "phi4_mm":
        # Phi-4 MM: prefer SDPA to avoid flash-attn, and avoid meta init issues.
        common_kwargs["attn_implementation"] = "sdpa"
        common_kwargs["low_cpu_mem_usage"] = False
    if spec.family in {"phi3_vision", "phi4_mm"}:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=True)
            if spec.family == "phi3_vision":
                setattr(config, "attn_implementation", "eager")
                setattr(config, "_attn_implementation", "eager")
            else:
                setattr(config, "attn_implementation", "sdpa")
                setattr(config, "_attn_implementation", "sdpa")
            common_kwargs["config"] = config
        except Exception:
            pass

    if spec.family == "emu3":
        # Emu3 is large; use device_map to reduce peak memory where possible.
        common_kwargs.setdefault("low_cpu_mem_usage", True)
        # Default to CPU to avoid GPU OOM, then move only vqmodel if possible.
        common_kwargs.setdefault("device_map", {"": "cpu"})
        common_kwargs.setdefault("offload_state_dict", True)
        common_kwargs.setdefault("offload_folder", "offload_emu3")

    # Some models (e.g., Phi-4 MM) fail under meta init. Temporarily disable meta init when needed.
    restore_init_ctx = None
    restore_tied_keys = None
    restore_tie_weights = None
    restore_mark_tied = None
    restore_move_missing = None
    restore_peft_get = None
    restore_peft_get_mapping = None
    if spec.family in {"phi4_mm"}:
        try:
            from transformers.modeling_utils import PreTrainedModel, local_torch_dtype

            restore_init_ctx = PreTrainedModel.get_init_context
            restore_tied_keys = PreTrainedModel.get_expanded_tied_weights_keys
            restore_tie_weights = PreTrainedModel.tie_weights
            restore_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized
            restore_move_missing = PreTrainedModel._move_missing_keys_from_meta_to_device

            def _no_meta_init(cls, dtype, is_quantized, _is_ds_init_called):
                # Avoid meta device for models that call .item() during init.
                return [local_torch_dtype(dtype, cls.__name__)]

            PreTrainedModel.get_init_context = classmethod(_no_meta_init)

            def _safe_get_expanded_tied_weights_keys(self, *args, **kwargs):
                try:
                    return restore_tied_keys(self, *args, **kwargs)
                except Exception:
                    # Phi-4 MM defines tied weights in a format incompatible with this transformers version.
                    return []

            PreTrainedModel.get_expanded_tied_weights_keys = _safe_get_expanded_tied_weights_keys

            def _safe_tie_weights(self, *args, **kwargs):
                try:
                    return restore_tie_weights(self, *args, **kwargs)
                except Exception:
                    return

            PreTrainedModel.tie_weights = _safe_tie_weights

            def _safe_mark_tied(self, *args, **kwargs):
                try:
                    return restore_mark_tied(self, *args, **kwargs)
                except Exception:
                    return

            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

            def _safe_move_missing(self, missing_keys, *args, **kwargs):
                try:
                    return restore_move_missing(self, missing_keys, *args, **kwargs)
                except Exception:
                    return

            PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move_missing

            # Patch peft.get_peft_model to add missing prepare_inputs_for_generation
            try:
                import peft

                restore_peft_get = peft.get_peft_model
                try:
                    from peft import mapping_func as _peft_mapping

                    restore_peft_get_mapping = _peft_mapping.get_peft_model
                except Exception:
                    restore_peft_get_mapping = None

                def _patched_get_peft_model(model, *args, **kwargs):
                    if not hasattr(model, "prepare_inputs_for_generation"):
                        def _prepare_inputs_for_generation(*_args, **_kwargs):
                            return {}
                        model.prepare_inputs_for_generation = _prepare_inputs_for_generation
                    return restore_peft_get(model, *args, **kwargs)

                peft.get_peft_model = _patched_get_peft_model
                if restore_peft_get_mapping is not None:
                    _peft_mapping.get_peft_model = _patched_get_peft_model
            except Exception:
                restore_peft_get = None
                restore_peft_get_mapping = None
        except Exception:
            restore_init_ctx = None
            restore_tied_keys = None
            restore_tie_weights = None
            restore_mark_tied = None
            restore_move_missing = None
            restore_peft_get = None
            restore_peft_get_mapping = None

    try:
        # Try specific classes first for better compatibility
        try:
            if spec.family == "qwen3_vl":
                from transformers import AutoModelForImageTextToText

                model = AutoModelForImageTextToText.from_pretrained(
                    spec.hf_id, **common_kwargs
                )
            elif spec.family == "siglip2":
                model = AutoModel.from_pretrained(spec.hf_id, **common_kwargs)
            elif spec.family == "kosmos2_5":
                from transformers import Kosmos2_5ForConditionalGeneration

                model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                    spec.hf_id, **common_kwargs
                )
            elif spec.family == "phi3_vision":
                # Use AutoModelForCausalLM with remote code for Phi-3 Vision.
                model = None
            elif spec.family == "phi4_mm":
                # Use AutoModelForCausalLM with remote code for Phi-4 MM.
                model = None
            elif spec.family == "llama3_2_vision":
                from transformers import MllamaForConditionalGeneration

                model = MllamaForConditionalGeneration.from_pretrained(
                    spec.hf_id, **common_kwargs
                )
            elif spec.family == "emu3":
                from transformers import Emu3ForConditionalGeneration

                model = Emu3ForConditionalGeneration.from_pretrained(
                    spec.hf_id, **common_kwargs
                )
        except Exception:
            model = None

        if model is None and spec.family == "siglip2":
            model = AutoModel.from_pretrained(spec.hf_id, **common_kwargs)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id, **common_kwargs
            )
    finally:
        if restore_init_ctx is not None:
            try:
                from transformers.modeling_utils import PreTrainedModel

                PreTrainedModel.get_init_context = restore_init_ctx
            except Exception:
                pass
        if restore_tied_keys is not None:
            try:
                from transformers.modeling_utils import PreTrainedModel

                PreTrainedModel.get_expanded_tied_weights_keys = restore_tied_keys
            except Exception:
                pass
        if restore_tie_weights is not None:
            try:
                from transformers.modeling_utils import PreTrainedModel

                PreTrainedModel.tie_weights = restore_tie_weights
            except Exception:
                pass
        if restore_mark_tied is not None:
            try:
                from transformers.modeling_utils import PreTrainedModel

                PreTrainedModel.mark_tied_weights_as_initialized = restore_mark_tied
            except Exception:
                pass
        if restore_move_missing is not None:
            try:
                from transformers.modeling_utils import PreTrainedModel

                PreTrainedModel._move_missing_keys_from_meta_to_device = restore_move_missing
            except Exception:
                pass
        if restore_peft_get is not None:
            try:
                import peft

                peft.get_peft_model = restore_peft_get
                if restore_peft_get_mapping is not None:
                    from peft import mapping_func as _peft_mapping

                    _peft_mapping.get_peft_model = restore_peft_get_mapping
            except Exception:
                pass

    model.eval()
    if getattr(model, "hf_device_map", None) is None and spec.family != "emu3":
        model.to(device)
    return model, processor


def extract_embeddings(
    model_key: str,
    model,
    processor,
    image_paths: List[str],
    device: str,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if model_key == "dinov3":
        # Vision-only DINOv3: use last_hidden_state patch tokens (exclude CLS).
        embeddings: List[torch.Tensor] = []
        try:
            dinov3_dtype = next(model.parameters()).dtype
        except Exception:
            dinov3_dtype = torch.float32
        for i in range(0, len(image_paths), batch_size):
            images = prepare_images(image_paths[i : i + batch_size])
            inputs = processor(images=images, return_tensors="pt")
            inputs = move_to_device(inputs, device, dinov3_dtype)
            with torch.no_grad():
                outputs = model(**inputs)
            hidden = _get_last_hidden_state(outputs)
            # DINOv3 may include CLS and register tokens; keep only patch tokens.
            if hidden.dim() == 3:
                num_register = 0
                cfg = getattr(model, "config", None)
                if cfg is not None:
                    num_register = getattr(cfg, "num_register_tokens", None)
                    if num_register is None:
                        num_register = getattr(cfg, "num_registers", 0)
                try:
                    num_register = int(num_register or 0)
                except Exception:
                    num_register = 0
                start = 1 + max(0, num_register)
                if hidden.size(1) > start:
                    pooled = hidden[:, start:, :].mean(dim=1)
                    pooled = F.normalize(pooled, dim=-1)
                else:
                    pooled = _pool_tokens(hidden)
            else:
                pooled = _pool_tokens(hidden)
            embeddings.append(pooled.detach().cpu())
        return torch.cat(embeddings, dim=0)

    if model_key == "emu3":
        # Emu3 uses a VQ-VAE vision backbone. Use encoder outputs (pre-LLM) as vision-only embeddings.
        vqmodel = None
        if hasattr(model, "model") and hasattr(model.model, "vqmodel"):
            vqmodel = model.model.vqmodel
        elif hasattr(model, "vqmodel"):
            vqmodel = model.vqmodel
        if vqmodel is None:
            raise RuntimeError("Could not locate Emu3 VQ model for vision embeddings")

        emu_device = device
        emu_dtype = dtype
        try:
            p = next(vqmodel.parameters())
            emu_device = p.device
            emu_dtype = p.dtype
        except StopIteration:
            pass
        except Exception:
            pass

        if not _is_cuda_device(emu_device) and emu_dtype in (torch.float16, torch.bfloat16):
            # CPU float16/bfloat16 ops may be unsupported; promote vision backbone to float32.
            try:
                vqmodel.to(dtype=torch.float32)
                emu_dtype = torch.float32
            except Exception:
                pass

        embeddings: List[torch.Tensor] = []
        emu_batch = min(batch_size, 1) if _is_cuda_device(emu_device) else batch_size
        for i in range(0, len(image_paths), emu_batch):
            images = prepare_images(image_paths[i : i + emu_batch])
            if hasattr(processor, "image_processor"):
                inputs = processor.image_processor(images=images, return_tensors="pt")
            else:
                inputs = processor(images=images, return_tensors="pt")

            multi_crop = False
            num_crops = 1
            if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
                if inputs["pixel_values"].dim() == 5:
                    multi_crop = True
                    b, n, c, h, w = inputs["pixel_values"].shape
                    num_crops = n
                    inputs["pixel_values"] = inputs["pixel_values"].view(b * n, c, h, w)

            pixel_values = inputs["pixel_values"].to(device=emu_device, dtype=emu_dtype)
            h, w = pixel_values.shape[-2:]
            image_sizes = torch.tensor([[h, w]] * pixel_values.shape[0], device=emu_device)

            with torch.no_grad():
                outputs = vqmodel.encode(pixel_values, image_sizes=image_sizes)

            hidden = outputs.last_hidden_state
            if hidden.dim() == 5:
                # hidden: [B, T, C, H, W] -> mean over T/H/W to get [B, C]
                pooled = hidden.mean(dim=(1, 3, 4))
            elif hidden.dim() == 4:
                # [B, C, H, W]
                pooled = hidden.mean(dim=(2, 3))
            else:
                pooled = hidden

            pooled = F.normalize(pooled, dim=-1)

            if multi_crop:
                pooled = pooled.view(-1, num_crops, pooled.size(-1)).mean(dim=1)
                pooled = F.normalize(pooled, dim=-1)

            embeddings.append(pooled.detach().cpu())

        return torch.cat(embeddings, dim=0)

    if model_key == "siglip2":
        # SigLIP2: use pooled image embeddings from the dual-tower model.
        embeddings: List[torch.Tensor] = []
        for i in range(0, len(image_paths), batch_size):
            images = prepare_images(image_paths[i : i + batch_size])

            if hasattr(processor, "image_processor"):
                inputs = processor.image_processor(images=images, return_tensors="pt")
            else:
                inputs = processor(images=images, return_tensors="pt")

            multi_crop = False
            num_crops = 1
            if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
                if inputs["pixel_values"].dim() == 5:
                    multi_crop = True
                    b, n, c, h, w = inputs["pixel_values"].shape
                    num_crops = n
                    inputs["pixel_values"] = inputs["pixel_values"].view(b * n, c, h, w)

            inputs = move_to_device(inputs, device, dtype)
            outputs = None
            pooled = None

            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    kwargs = _filter_kwargs_for_callable(model.get_image_features, inputs)
                    pooled = model.get_image_features(**kwargs)
                else:
                    kwargs = _filter_kwargs_for_module(model, inputs)
                    outputs = model(**kwargs)

            pooled = _extract_image_embedding(pooled)
            if pooled is None:
                pooled = _extract_image_embedding(outputs)

            if pooled is None:
                raise RuntimeError("SigLIP2 outputs missing image embeddings")

            pooled = F.normalize(pooled, dim=-1)

            if multi_crop:
                pooled = pooled.view(-1, num_crops, pooled.size(-1)).mean(dim=1)
                pooled = F.normalize(pooled, dim=-1)

            embeddings.append(pooled.detach().cpu())

        return torch.cat(embeddings, dim=0)

    vision_model = get_vision_module(model, model_key)
    if vision_model is None:
        raise RuntimeError("Could not locate a vision encoder module for this model")

    embeddings: List[torch.Tensor] = []

    for i in range(0, len(image_paths), batch_size):
        images = prepare_images(image_paths[i : i + batch_size])

        # Prefer image_processor when available to avoid text requirements.
        if model_key == "qwen3_vl":
            # Qwen3-VL processor expects text; provide empty prompts to emit image_grid_thw.
            inputs = processor(text=[""] * len(images), images=images, return_tensors="pt")
        elif hasattr(processor, "image_processor"):
            inputs = processor.image_processor(images=images, return_tensors="pt")
        else:
            inputs = processor(images=images, return_tensors="pt")

        # Some processors (e.g., Phi-4 MM) return input_image_embeds instead of pixel_values.
        if "pixel_values" not in inputs and "input_image_embeds" in inputs:
            inputs["pixel_values"] = inputs["input_image_embeds"]

        # Track multi-crop if present: [B, N, C, H, W]
        multi_crop = False
        num_crops = 1
        if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
            if inputs["pixel_values"].dim() == 5:
                multi_crop = True
                b, n, c, h, w = inputs["pixel_values"].shape
                num_crops = n
                inputs["pixel_values"] = inputs["pixel_values"].view(b * n, c, h, w)

        inputs = move_to_device(inputs, device, dtype)
        kwargs = _filter_kwargs_for_module(vision_model, inputs)

        with torch.no_grad():
            try:
                if model_key == "qwen3_vl":
                    # Qwen3-VL vision encoder may expect hidden_states + grid_thw (like Qwen2.5-VL).
                    pixel_values = inputs.get("pixel_values", None)
                    grid_thw = inputs.get("image_grid_thw", None)
                    if grid_thw is None:
                        grid_thw = inputs.get("grid_thw", None)
                    if grid_thw is None and pixel_values is not None:
                        patch_size = None
                        if hasattr(model, "config") and hasattr(model.config, "vision_config"):
                            patch_size = getattr(model.config.vision_config, "patch_size", None)
                        if patch_size is None:
                            patch_size = getattr(vision_model, "patch_size", None)
                        if patch_size is None and hasattr(vision_model, "config"):
                            patch_size = getattr(vision_model.config, "patch_size", None)
                        if isinstance(patch_size, (list, tuple)):
                            patch_size = patch_size[0]
                        if patch_size is not None:
                            h = pixel_values.shape[-2] // patch_size
                            w = pixel_values.shape[-1] // patch_size
                            grid_thw = torch.tensor([[1, h, w]] * pixel_values.shape[0], device=pixel_values.device)
                            inputs["image_grid_thw"] = grid_thw
                            inputs["grid_thw"] = grid_thw
                    if pixel_values is None or grid_thw is None:
                        raise RuntimeError("Qwen3-VL requires pixel_values and image_grid_thw/grid_thw")
                    try:
                        outputs = vision_model(hidden_states=pixel_values, grid_thw=grid_thw, return_dict=True)
                    except Exception:
                        # Some Qwen3 vision towers expect patch-embedded tokens.
                        if hasattr(vision_model, "patch_embed"):
                            hidden_states = vision_model.patch_embed(pixel_values)
                            outputs = vision_model(hidden_states=hidden_states, grid_thw=grid_thw, return_dict=True)
                        else:
                            outputs = vision_model(pixel_values=pixel_values, grid_thw=grid_thw, return_dict=True)
                else:
                    outputs = vision_model(**kwargs)
            except Exception:
                # Fall back to minimal pixel_values if filtering failed.
                if "pixel_values" in inputs:
                    outputs = vision_model(pixel_values=inputs["pixel_values"])
                elif "image_pixel_values" in inputs:
                    outputs = vision_model(image_pixel_values=inputs["image_pixel_values"])
                else:
                    raise

        # Model-specific embedding extraction
        if model_key == "qwen3_vl":
            # Qwen3-VL: use vision-only outputs before LLM fusion.
            # Prefer pooler_output from the vision merger (still vision-only, pre-LLM).
            grid_thw = inputs.get("image_grid_thw", None)
            if grid_thw is None:
                grid_thw = inputs.get("grid_thw", None)
            if grid_thw is None:
                raise RuntimeError("Qwen3-VL requires image_grid_thw/grid_thw for pooling")

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                tokens = outputs.pooler_output
                # If tokens are flattened [sum_tokens, dim], split per image using merged grid.
                if tokens.dim() == 2:
                    spatial_merge = getattr(model.config.vision_config, "spatial_merge_size", 1)
                    lengths = (
                        grid_thw[:, 0]
                        * (grid_thw[:, 1] // spatial_merge)
                        * (grid_thw[:, 2] // spatial_merge)
                    ).tolist()
                    splits = torch.split(tokens, lengths, dim=0)
                    pooled = torch.stack([s.mean(dim=0) for s in splits], dim=0)
                else:
                    pooled = tokens.mean(dim=1)
            else:
                hidden = _get_last_hidden_state(outputs)
                if hidden.dim() == 2:
                    lengths = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
                    splits = torch.split(hidden, lengths, dim=0)
                    pooled = torch.stack([s.mean(dim=0) for s in splits], dim=0)
                else:
                    pooled = _pool_tokens(hidden)

            pooled = F.normalize(pooled, dim=-1)
        else:
            # Default: mean-pool patch tokens, exclude CLS if present.
            hidden = _get_last_hidden_state(outputs)
            pooled = _pool_tokens(hidden)

        if multi_crop:
            # Average across crops to get one embedding per original image.
            pooled = pooled.view(-1, num_crops, pooled.size(-1)).mean(dim=1)
            pooled = F.normalize(pooled, dim=-1)

        embeddings.append(pooled.detach().cpu())

    return torch.cat(embeddings, dim=0)


def compute_pairwise_stats(embeddings: torch.Tensor) -> Tuple[float, float, int, np.ndarray]:
    E = F.normalize(embeddings, dim=-1)
    S = E @ E.T
    idx = torch.triu_indices(E.size(0), E.size(0), offset=1)
    sims = S[idx[0], idx[1]]
    mean = sims.mean().item()
    std = sims.std(unbiased=False).item()
    return mean, std, sims.numel(), sims.cpu().numpy()


def sanity_checks(model_name: str, embeddings: torch.Tensor, mean_cos: float) -> None:
    if embeddings.shape[0] < 2:
        _warn(f"{model_name}: fewer than 2 embeddings; stats may be meaningless")
        return

    # Check if embeddings are identical to the first vector
    diffs = (embeddings[1:] - embeddings[0]).abs().max(dim=1).values
    if torch.all(diffs < 1e-6):
        _warn(f"{model_name}: all embeddings appear identical")

    # Per-dimension variance
    per_dim_std = embeddings.std(dim=0)
    mean_std = per_dim_std.mean().item()
    if mean_std < 1e-4:
        _warn(f"{model_name}: per-dimension variance is near zero (mean std={mean_std:.2e})")

    if mean_cos > 0.99:
        _warn(f"{model_name}: mean cosine similarity is very high ({mean_cos:.4f})")


def format_table(rows: List[Dict[str, str]]) -> str:
    headers = ["ModelName", "HF_ID", "EmbDim", "NumImages", "NumPairs", "MeanCos", "StdCos", "TimeSec"]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    def fmt_row(row: Dict[str, str]) -> str:
        return " | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)

    sep = "-+-".join("-" * col_widths[h] for h in headers)
    lines = [fmt_row({h: h for h in headers}), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def safe_model_name(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding geometry for new VLMs.")
    parser.add_argument("--data_root", required=True, help="Root directory containing small_city dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistency")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/compare_embeddings_siglip2",
        help="Directory to write results and sims_*.npy",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names, families, or HF IDs to run (case-insensitive).",
    )
    args = parser.parse_args()

    set_seeds(args.seed)
    device, dtype = choose_device(args.device)

    data_root = Path(args.data_root)
    if not (data_root / "small_city").exists():
        raise FileNotFoundError(f"Expected small_city under {data_root}")

    selected_list = Path(f"selected_images_seed{args.seed}.txt")
    image_paths = load_selected_images(selected_list)

    print(f"Using device={device}, dtype={dtype}")
    print(f"Loaded {len(image_paths)} images from {selected_list}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        tokens = [t.strip().lower() for t in args.models.split(",") if t.strip()]
        selected_specs = [
            spec
            for spec in MODEL_SPECS
            if any(
                token in spec.name.lower()
                or token in spec.family.lower()
                or token in spec.hf_id.lower()
                for token in tokens
            )
        ]
        if not selected_specs:
            raise ValueError(
                f"No models matched --models={args.models}. "
                f"Available: {', '.join(spec.name for spec in MODEL_SPECS)}"
            )
    else:
        selected_specs = MODEL_SPECS

    results = []
    table_rows = []

    for spec in selected_specs:
        start = time.time()
        try:
            print(f"\nLoading {spec.name} ({spec.hf_id})...")
            model, processor = load_model_and_processor(spec, device, dtype)
            embeddings = extract_embeddings(
                spec.family, model, processor, image_paths, device, args.batch_size, dtype
            )
            emb_dim = embeddings.shape[1]
            mean, std, num_pairs, sims = compute_pairwise_stats(embeddings)

            sanity_checks(spec.name, embeddings, mean)

            model_key = safe_model_name(spec.name)
            np.save(output_dir / f"sims_{model_key}.npy", sims)

            time_sec = time.time() - start
            result = {
                "ModelName": spec.name,
                "HF_ID": spec.hf_id,
                "EmbDim": emb_dim,
                "NumImages": embeddings.shape[0],
                "NumPairs": num_pairs,
                "MeanCos": mean,
                "StdCos": std,
                "TimeSec": time_sec,
            }
            results.append(result)
            table_rows.append({
                "ModelName": spec.name,
                "HF_ID": spec.hf_id,
                "EmbDim": str(emb_dim),
                "NumImages": str(embeddings.shape[0]),
                "NumPairs": str(num_pairs),
                "MeanCos": f"{mean:.6f}",
                "StdCos": f"{std:.6f}",
                "TimeSec": f"{time_sec:.2f}",
            })
            print(f"{spec.name}: emb_dim={emb_dim}, mean={mean:.6f}, std={std:.6f}")
        except Exception as e:
            print(f"Warning: Skipping {spec.name} due to error: {e}")
        finally:
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    (output_dir / "results_new_models_extended.json").write_text(json.dumps(results, indent=2))
    if table_rows:
        print("\nResults:")
        print(format_table(table_rows))
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
