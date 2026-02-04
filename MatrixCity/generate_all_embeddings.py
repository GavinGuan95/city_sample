#!/usr/bin/env python3
"""Generate DINOv3 embeddings for all image folders under a root.

Default behavior:
- Recursively discover all folders containing images under --root.
- For each folder, write outputs to <folder>/<model>/ (e.g., dinov3).

Outputs per folder:
- embeddings.npy (or shards if large)
- paths.txt
- shape.json
- failed.txt (optional, only if any failures)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import inspect
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DINOv3_HF_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
SIGLIP2_HF_ID = "google/siglip2-so400m-patch16-naflex"


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def enable_tf32_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


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


def _pool_dinov3(model, hidden: torch.Tensor) -> torch.Tensor:
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
    return pooled


def _filter_kwargs_for_module(module, inputs: dict) -> dict:
    try:
        sig = inspect.signature(module.forward)
        params = sig.parameters
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return inputs
        allowed = set(params.keys())
    except Exception:
        return inputs

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


def _filter_kwargs_for_callable(func, inputs: dict) -> dict:
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


def move_to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype, non_blocking=True)
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def load_dinov3(device: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINOv3_HF_ID)
    model = AutoModel.from_pretrained(DINOv3_HF_ID, torch_dtype=torch.float32)
    model.eval().to(device)
    return model, processor, torch.float32


def load_siglip2(device: str):
    from transformers import AutoModel, AutoProcessor

    dtype = torch.float16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(SIGLIP2_HF_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(SIGLIP2_HF_ID, torch_dtype=dtype, trust_remote_code=True)
    model.eval().to(device)
    return model, processor, dtype


def infer_embedding_dim(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "embed_dim"):
        return int(cfg.embed_dim)
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None:
        if hasattr(vision_cfg, "hidden_size"):
            return int(vision_cfg.hidden_size)
        if hasattr(vision_cfg, "embed_dim"):
            return int(vision_cfg.embed_dim)
    return None


def compute_embeddings(
    model_key: str,
    model,
    processor,
    images: List[Image.Image],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if model_key == "dinov3":
        inputs = processor(images=images, return_tensors="pt")
        inputs = move_to_device(inputs, device, dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        hidden = _get_last_hidden_state(outputs)
        pooled = _pool_dinov3(model, hidden)
        return pooled.detach().cpu()

    if model_key == "siglip2":
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
        with torch.inference_mode():
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

        return pooled.detach().cpu()

    raise ValueError(f"Unsupported model: {model_key}")


class ImageFolderDataset(Dataset):
    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
            return idx, img, str(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {path}") from exc


def collate_batch(batch):
    idxs, images, paths = [], [], []
    for idx, img, path in batch:
        idxs.append(idx)
        images.append(img)
        paths.append(path)
    return idxs, images, paths


def discover_image_folders(root: Path, exts: set[str]) -> List[Path]:
    folders: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        has_image = any(Path(name).suffix.lower() in exts for name in filenames)
        if has_image:
            folders.append(Path(dirpath))
    folders.sort()
    return folders


def list_images_in_folder(folder: Path, exts: set[str]) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort()
    return paths


class EmbeddingWriter:
    def __init__(
        self,
        out_dir: Path,
        total: int,
        dim: int,
        dtype: np.dtype,
        shard_size: int,
    ) -> None:
        self.out_dir = out_dir
        self.total = total
        self.dim = dim
        self.dtype = dtype
        self.shard_size = shard_size
        self.shards: Optional[List[Tuple[int, int, np.memmap, Path]]] = None
        self.single: Optional[np.memmap] = None

        if shard_size > 0 and total > shard_size:
            num_shards = math.ceil(total / shard_size)
            shards: List[Tuple[int, int, np.memmap, Path]] = []
            for i in range(num_shards):
                start = i * shard_size
                end = min(total, (i + 1) * shard_size)
                path = out_dir / f"embeddings_{i:03d}.npy"
                mem = np.lib.format.open_memmap(
                    path, mode="w+", dtype=dtype, shape=(end - start, dim)
                )
                shards.append((start, end, mem, path))
            self.shards = shards
        else:
            path = out_dir / "embeddings.npy"
            self.single = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=(total, dim)
            )

    def write_contiguous(self, start_idx: int, batch_array: np.ndarray) -> None:
        if self.single is not None:
            self.single[start_idx : start_idx + len(batch_array)] = batch_array
            return
        assert self.shards is not None
        remaining = len(batch_array)
        offset = 0
        cur = start_idx
        while remaining > 0:
            shard_idx = cur // self.shard_size
            shard_start, shard_end, shard_mem, _ = self.shards[shard_idx]
            shard_offset = cur - shard_start
            shard_capacity = shard_end - shard_start - shard_offset
            write_len = min(remaining, shard_capacity)
            shard_mem[shard_offset : shard_offset + write_len] = batch_array[
                offset : offset + write_len
            ]
            cur += write_len
            offset += write_len
            remaining -= write_len

    def flush(self) -> None:
        if self.single is not None:
            self.single.flush()
        if self.shards is not None:
            for _, _, mem, _ in self.shards:
                mem.flush()

    def shard_metadata(self) -> Optional[List[dict]]:
        if self.shards is None:
            return None
        data = []
        for start, end, _, path in self.shards:
            data.append({"start": start, "end": end, "path": path.name})
        return data


def process_folder(
    folder: Path,
    root: Path,
    out_dir: Path,
    model_key: str,
    model_id: str,
    model,
    processor,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    shard_size: int,
    quick_count: Optional[int] = None,
) -> None:
    images = list_images_in_folder(folder, SUPPORTED_EXTS)
    if not images:
        return
    if quick_count is not None:
        images = images[:quick_count]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[folder] {folder} -> {out_dir} ({len(images)} images)")

    paths_rel = []
    for p in images:
        try:
            paths_rel.append(str(p.relative_to(root)))
        except Exception:
            paths_rel.append(str(p))
    (out_dir / "paths.txt").write_text("\n".join(paths_rel) + "\n")

    dataset = ImageFolderDataset(images)
    pin_memory = device.startswith("cuda") and torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    embedding_dim = infer_embedding_dim(model)
    writer: Optional[EmbeddingWriter] = None
    if embedding_dim is not None:
        writer = EmbeddingWriter(out_dir, len(images), embedding_dim, np.float32, shard_size)

    write_pos = 0
    for idxs, batch_images, _batch_paths in loader:
        batch_count = len(idxs)
        ok_images = batch_images
        pooled = compute_embeddings(
            model_key=model_key,
            model=model,
            processor=processor,
            images=ok_images,
            device=device,
            dtype=dtype,
        )
        pooled_np = pooled.numpy()

        if writer is None:
            embedding_dim = pooled_np.shape[1]
            writer = EmbeddingWriter(out_dir, len(images), embedding_dim, np.float32, shard_size)

        if pooled_np.shape[0] != batch_count:
            raise RuntimeError(
                f"Unexpected batch size from model: got {pooled_np.shape[0]}, expected {batch_count}"
            )

        if idxs[0] != write_pos or idxs[-1] != write_pos + batch_count - 1:
            raise RuntimeError("DataLoader returned non-contiguous batch ordering")

        writer.write_contiguous(write_pos, pooled_np.astype(np.float32, copy=False))
        write_pos += batch_count

    if writer is None:
        raise RuntimeError(f"No valid images decoded in {folder}")

    writer.flush()

    meta = {
        "num_images": len(images),
        "embedding_dim": int(embedding_dim),
        "dtype": "float32",
        "model_key": model_key,
        "model_id": model_id,
    }
    (out_dir / "shape.json").write_text(json.dumps(meta, indent=2) + "\n")

    shards = writer.shard_metadata()
    if shards is not None:
        (out_dir / "shards.json").write_text(json.dumps(shards, indent=2) + "\n")

    # Any decode or model errors should have raised earlier.


def load_model_and_processor(model_key: str, device: str):
    if model_key == "dinov3":
        model, processor, dtype = load_dinov3(device)
        return model, processor, dtype, DINOv3_HF_ID
    if model_key == "siglip2":
        model, processor, dtype = load_siglip2(device)
        return model, processor, dtype, SIGLIP2_HF_ID
    raise ValueError(f"Unsupported model: {model_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov3", choices=["dinov3", "siglip2"])
    parser.add_argument("--root", default="small_city", help="Root to scan for image folders.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output base directory. If unset, writes to <folder>/<model>/.",
    )
    parser.add_argument(
        "--discover-folders",
        type=_parse_bool,
        default=True,
        help="Discover all folders under --root that contain images.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--store-dtype", default="float32", choices=["float32"])
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick-folder",
        default=None,
        help="Run a quick test on a specific folder (disables discovery).",
    )
    parser.add_argument(
        "--quick-name",
        default=None,
        help="Output folder name for quick mode (default: <model>_quick).",
    )
    parser.add_argument(
        "--quick-count",
        type=int,
        default=100,
        help="Number of images to embed in quick mode.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_base = Path(args.out).resolve() if args.out else None

    device = choose_device(args.device)
    enable_tf32_if_cuda(device)
    print(f"[device] {device}")

    model, processor, dtype, model_id = load_model_and_processor(args.model, device)

    quick_out_dir: Optional[Path] = None
    quick_name: Optional[str] = None
    if args.quick_folder:
        quick_folder = Path(args.quick_folder).resolve()
        if not quick_folder.exists():
            raise FileNotFoundError(f"Quick folder not found: {quick_folder}")
        folders = [quick_folder]
        quick_name = args.quick_name or f"{args.model}_quick"
        if args.out:
            quick_out_dir = Path(args.out).resolve() / quick_name
        else:
            quick_out_dir = quick_folder / quick_name
        print(f"[quick] {quick_folder} -> {quick_out_dir} (count={args.quick_count})")
    elif args.discover_folders:
        folders = discover_image_folders(root, SUPPORTED_EXTS)
    else:
        folders = [root]

    if not folders:
        raise RuntimeError(f"No image folders found under {root}")
    print(f"[folders] {len(folders)}")

    for folder in folders:
        if args.quick_folder:
            assert quick_out_dir is not None
            out_dir = quick_out_dir
        else:
            out_dir = (folder / args.model) if out_base is None else (out_base / folder.relative_to(root))
        if out_dir.exists() and not args.overwrite:
            print(f"[skip] {out_dir} exists (use --overwrite to regenerate)")
            continue
        process_folder(
            folder=folder,
            root=root,
            out_dir=out_dir,
            model_key=args.model,
            model_id=model_id,
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shard_size=args.shard_size,
            quick_count=args.quick_count if args.quick_folder else None,
        )


if __name__ == "__main__":
    main()
