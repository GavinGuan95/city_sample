#!/usr/bin/env python3
"""
Compare visual embedding geometry across multiple vision-language models.
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    family: str
    optional: bool = False


MODEL_SPECS = [
    ModelSpec("Qwen2.5-VL", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
    ModelSpec("LLaVA-1.6", "liuhaotian/llava-v1.6-vicuna-7b", "llava"),
    ModelSpec("Idefics2", "HuggingFaceM4/idefics2-8b", "idefics2"),
    ModelSpec("PaliGemma", "google/paligemma-3b-mix-224", "paligemma"),
    ModelSpec("DINOv2", "facebook/dinov2-giant", "dinov2", optional=True),
]


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


def list_images(data_root: str) -> List[str]:
    root = Path(data_root) / "small_city"
    if not root.exists():
        raise FileNotFoundError(f"Expected dataset at {root}")
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in SUPPORTED_EXTS:
                paths.append(str(Path(dirpath) / name))
    return paths


def _read_cached_list(cache_path: Path) -> Optional[List[str]]:
    if not cache_path.exists():
        return None
    lines = [line.strip() for line in cache_path.read_text().splitlines() if line.strip()]
    return lines if lines else None


def select_images(
    all_paths: List[str],
    seed: int,
    k: int = 100,
    cache_path: Optional[Path] = None,
) -> List[str]:
    if cache_path is None:
        cache_path = Path(f"selected_images_seed{seed}.txt")

    cached = _read_cached_list(cache_path)
    if cached is not None:
        if len(cached) == k and all(Path(p).exists() for p in cached):
            return cached
        # Cache is stale or incomplete; fall back to resampling.

    if len(all_paths) < k:
        raise ValueError(f"Need at least {k} images, found {len(all_paths)}")

    rng = random.Random(seed)
    all_paths_sorted = sorted(all_paths)
    selected = rng.sample(all_paths_sorted, k)
    cache_path.write_text("\n".join(selected) + "\n")
    return selected


def choose_device(device_arg: str) -> Tuple[str, torch.dtype]:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if device.startswith("cuda") and torch.cuda.is_available():
        # Prefer float16 for broad model/operator compatibility on GPU.
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


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


def prepare_vision_inputs(processor, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
    # Most HF processors accept images directly. If not, fall back to image_processor.
    try:
        inputs = processor(images=images, return_tensors="pt")
    except Exception:
        if hasattr(processor, "image_processor"):
            inputs = processor.image_processor(images=images, return_tensors="pt")
        else:
            raise
    if not isinstance(inputs, dict):
        # Some processors return BatchFeature; treat as dict.
        inputs = dict(inputs)
    return inputs


def _get_submodule(model, path: str):
    cur = model
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def get_vision_model(model, family: str):
    # Family-specific hooks to obtain the vision encoder BEFORE any projector/LLM mixing.
    if family == "llava":
        # LLaVA: use CLIP vision tower outputs (pre-projection).
        if hasattr(model, "get_vision_tower"):
            vt = model.get_vision_tower()
            if isinstance(vt, list):
                vt = vt[0]
            if hasattr(vt, "vision_tower"):
                vt = vt.vision_tower
            if hasattr(vt, "vision_model"):
                vt = vt.vision_model
            return vt
        candidates = [
            "vision_tower",
            "model.vision_tower",
            "model.model.vision_tower",
        ]
    elif family == "qwen2_5_vl":
        # Qwen2.5-VL: use visual/vision_tower module before projector.
        candidates = [
            "visual",
            "vision_tower",
            "vision_model",
            "model.visual",
            "model.vision_tower",
            "model.vision_model",
            "model.model.visual",
            "model.model.vision_tower",
        ]
    elif family == "idefics2":
        # Idefics2: use SigLIP vision_model before any multimodal projection.
        candidates = [
            "vision_model",
            "model.vision_model",
            "model.model.vision_model",
            "vision_tower",
            "model.vision_tower",
        ]
    elif family == "paligemma":
        # PaliGemma: use SigLIP vision tower before projector.
        candidates = [
            "vision_tower",
            "vision_model",
            "model.vision_tower",
            "model.vision_model",
            "model.model.vision_tower",
            "model.model.vision_model",
        ]
    elif family == "dinov2":
        # DINOv2 is vision-only; the model itself is the vision encoder.
        return model
    else:
        candidates = []

    for path in candidates:
        sub = _get_submodule(model, path)
        if sub is not None:
            # unwrap potential wrappers
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
        # If signature inspection fails, pass everything.
        return inputs

    # Support alternate key names if needed.
    filtered: Dict[str, torch.Tensor] = dict(inputs)
    if "image_grid_thw" in filtered and "grid_thw" not in filtered:
        filtered["grid_thw"] = filtered["image_grid_thw"]
    if "grid_thw" in filtered and "image_grid_thw" not in filtered:
        filtered["image_grid_thw"] = filtered["grid_thw"]

    return {k: v for k, v in filtered.items() if k in allowed}


def _get_last_hidden_state(outputs) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise ValueError("Could not find last_hidden_state in vision outputs")


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


def load_model_and_processor(spec: ModelSpec, device: str, dtype: torch.dtype):
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoImageProcessor, AutoConfig
    try:
        from transformers import AutoModelForVision2Seq
    except Exception:
        AutoModelForVision2Seq = None

    if spec.family == "dinov2":
        processor = AutoImageProcessor.from_pretrained(spec.hf_id)
        model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=dtype)
        model.eval().to(device)
        return model, processor

    try:
        processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
    except Exception:
        # Some repos (e.g., LLaVA) may not ship a processor; fall back to vision tower image processor.
        cfg = AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=True)
        vision_id = getattr(cfg, "mm_vision_tower", None) or getattr(cfg, "vision_tower", None)
        if isinstance(vision_id, (list, tuple)):
            vision_id = vision_id[0]
        if vision_id is None:
            vision_id = "openai/clip-vit-large-patch14-336"
        processor = AutoImageProcessor.from_pretrained(vision_id)

    model = None
    if spec.family == "llava":
        try:
            from transformers import LlavaForConditionalGeneration

            model = LlavaForConditionalGeneration.from_pretrained(
                spec.hf_id, torch_dtype=dtype, trust_remote_code=True
            )
        except Exception:
            model = None
    elif spec.family == "idefics2":
        try:
            from transformers import Idefics2ForConditionalGeneration

            model = Idefics2ForConditionalGeneration.from_pretrained(
                spec.hf_id, torch_dtype=dtype, trust_remote_code=True
            )
        except Exception:
            model = None
    elif spec.family == "paligemma":
        try:
            from transformers import PaliGemmaForConditionalGeneration

            model = PaliGemmaForConditionalGeneration.from_pretrained(
                spec.hf_id, torch_dtype=dtype, trust_remote_code=True
            )
        except Exception:
            model = None
    elif spec.family == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                spec.hf_id, torch_dtype=dtype, trust_remote_code=True
            )
        except Exception:
            model = None

    if model is None:
        if AutoModelForVision2Seq is not None:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    spec.hf_id, torch_dtype=dtype, trust_remote_code=True
                )
            except Exception:
                model = None
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id, torch_dtype=dtype, trust_remote_code=True
            )

    model.eval().to(device)
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
    vision_model = get_vision_model(model, model_key)
    if vision_model is None:
        raise RuntimeError(f"Could not locate vision model for family '{model_key}'")

    embeddings: List[torch.Tensor] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images: List[Image.Image] = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))

        # Family-specific preprocessing
        if model_key == "idefics2" and hasattr(processor, "image_processor"):
            inputs = processor.image_processor(
                images=images, return_tensors="pt", do_image_splitting=False
            )
        else:
            inputs = prepare_vision_inputs(processor, images)

        inputs = move_to_device(inputs, device, dtype)

        with torch.no_grad():
            if model_key == "qwen2_5_vl":
                # Qwen2.5-VL vision encoder expects hidden_states (pixel values) + grid_thw.
                pixel_values = inputs.get("pixel_values", None)
                grid_thw = inputs.get("image_grid_thw", None)
                if grid_thw is None:
                    grid_thw = inputs.get("grid_thw", None)
                if pixel_values is None or grid_thw is None:
                    raise RuntimeError("Qwen2.5-VL requires pixel_values and image_grid_thw/grid_thw")
                outputs = vision_model(hidden_states=pixel_values, grid_thw=grid_thw, return_dict=True)
            elif model_key == "llava":
                pixel_values = inputs.get("pixel_values", None)
                if pixel_values is None:
                    raise RuntimeError("LLaVA requires pixel_values from the image processor")
                outputs = vision_model(pixel_values=pixel_values, return_dict=True)
            elif model_key in {"idefics2", "paligemma", "dinov2"}:
                pixel_values = inputs.get("pixel_values", None)
                if pixel_values is None:
                    raise RuntimeError("Model requires pixel_values from the image processor")
                if model_key == "idefics2" and pixel_values.dim() == 5:
                    # Idefics2 image processor returns [B, N, C, H, W]; flatten to a standard batch.
                    b, n, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.view(b * n, c, h, w)
                outputs = vision_model(pixel_values=pixel_values, return_dict=True)
            else:
                # Generic fallback: filter inputs for the vision tower.
                kwargs = _filter_kwargs_for_module(vision_model, inputs)
                outputs = vision_model(**kwargs)

        if model_key == "qwen2_5_vl":
            # Qwen2.5-VL: prefer pooler_output (post-merger) to avoid near-constant mean over pre-merge tokens.
            grid_thw = inputs.get("image_grid_thw", None)
            if grid_thw is None:
                grid_thw = inputs.get("grid_thw", None)
            if grid_thw is None:
                raise RuntimeError("Qwen2.5-VL requires image_grid_thw/grid_thw for pooling")
            pooled_tokens = outputs.pooler_output
            spatial_merge = getattr(model.config.vision_config, "spatial_merge_size", 1)
            lengths = (
                grid_thw[:, 0] * (grid_thw[:, 1] // spatial_merge) * (grid_thw[:, 2] // spatial_merge)
            ).tolist()
            splits = torch.split(pooled_tokens, lengths, dim=0)
            pooled = torch.stack([s.mean(dim=0) for s in splits], dim=0)
            pooled = F.normalize(pooled, dim=-1)
        else:
            hidden = _get_last_hidden_state(outputs)
            pooled = _pool_tokens(hidden)
        embeddings.append(pooled.detach().cpu())

    return torch.cat(embeddings, dim=0)


def compute_pairwise_stats(embeddings: torch.Tensor) -> Tuple[float, float, int, np.ndarray]:
    # embeddings should already be L2-normalized, but normalize again defensively.
    E = F.normalize(embeddings, dim=-1)
    S = E @ E.T
    idx = torch.triu_indices(E.size(0), E.size(0), offset=1)
    sims = S[idx[0], idx[1]]
    mean = sims.mean().item()
    std = sims.std(unbiased=False).item()
    return mean, std, sims.numel(), sims.cpu().numpy()


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
    parser = argparse.ArgumentParser(description="Compare visual embedding geometry across models.")
    parser.add_argument("--data_root", required=True, help="Root directory containing small_city dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/compare_embeddings",
        help="Directory to write results.json and sims_*.npy",
    )
    args = parser.parse_args()

    set_seeds(args.seed)
    device, dtype = choose_device(args.device)

    all_paths = list_images(args.data_root)
    selected = select_images(all_paths, seed=args.seed, k=100)

    print(f"Using device={device}, dtype={dtype}")
    print(f"Found {len(all_paths)} images; selected {len(selected)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    table_rows = []

    for spec in MODEL_SPECS:
        start = time.time()
        try:
            print(f"\nLoading {spec.name} ({spec.hf_id})...")
            model, processor = load_model_and_processor(spec, device, dtype)
            embeddings = extract_embeddings(
                spec.family, model, processor, selected, device, args.batch_size, dtype
            )
            emb_dim = embeddings.shape[1]
            mean, std, num_pairs, sims = compute_pairwise_stats(embeddings)

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
            msg = f"Skipping {spec.name} due to error: {e}"
            if spec.optional:
                print("Warning:", msg)
            else:
                print(msg)
        finally:
            # Free GPU memory between models.
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    if table_rows:
        print("\nResults:")
        print(format_table(table_rows))
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
