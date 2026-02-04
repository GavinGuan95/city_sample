**Goal**
Generate DINOv3-H+ embeddings for every image under `MatrixCity/small_city` and write them to `MatrixCity/small_city/street/test/small_city_road_down_test/dinov3`, with a focus on throughput and robustness. The design should also extend to SigLIP2 and `MatrixCity/big_city` later.

**Assumptions And Constraints**
- Use `facebook/dinov3-vith16plus-pretrain-lvd1689m` and the same preprocessing flow as `MatrixCity/compare_embeddings_new_models.py`.
- DINOv3-H+ is numerically sensitive; compute in float32 for stability, and store embeddings in float32 as requested.
- Output must be indexable by image path order. Strict determinism is not required.

**Proposed Script Interface**
- Script path: `MatrixCity/generate_all_embeddings.py` (new)
- CLI:
  - `--model dinov3` (default)
  - `--root MatrixCity/small_city`
  - `--out MatrixCity/small_city/street/test/small_city_road_down_test/dinov3`
  - `--discover-folders true` (default) to auto-discover all folders under `--root` that contain images and generate one embedding file per folder.
  - `--batch-size 64` (auto-tuned with a quick warmup)
  - `--num-workers 8`
  - `--device auto`
  - `--store-dtype float32` (fixed)
  - `--shard-size 50000` (default when sharding; max embeddings per file)

**Data Flow**
1. Discover folders under `--root` that contain images (default: any of `.png/.jpg/.jpeg/.webp`), and for each folder:
2. Enumerate image files inside the folder by extension and sort lexicographically for stable ordering.
3. Save the ordered list to `out/<folder_rel>/paths.txt` (one absolute or root-relative path per line).
4. Create a `torch.utils.data.Dataset` that loads image bytes and decodes to RGB.
5. Use a `DataLoader` with pinned memory and persistent workers to overlap CPU I/O and GPU inference.
6. Preprocess via `AutoImageProcessor` for DINOv3.
7. Run model under `torch.inference_mode()` and extract pooled patch embeddings as in `compare_embeddings_new_models.py`.
8. Write embeddings to disk incrementally using `numpy.lib.format.open_memmap` as `out/<folder_rel>/embeddings.npy`.

**Efficiency Tactics**
- I/O and decode:
  - Use multiple workers with `persistent_workers=True` and `prefetch_factor=2`.
  - Keep `pin_memory=True` and move tensors to GPU with `non_blocking=True`.
  - Consider `torchvision.io.read_image` + manual RGB conversion if PIL becomes a bottleneck.
- GPU utilization:
  - Use `torch.inference_mode()`.
  - Use a warmup pass to find a stable batch size for the current GPU.
  - Enable TF32 matmul for speed on RTX4090:
    - `torch.backends.cuda.matmul.allow_tf32 = True`
    - `torch.backends.cudnn.allow_tf32 = True`
    - `torch.set_float32_matmul_precision("high")`
- Pipeline overlap (optional, if needed):
  - Use a dedicated CUDA stream to prefetch next batch while the current batch runs.
  - This is only worth doing if GPU is underutilized after DataLoader tuning.
- Error handling:
  - Catch decode errors and log to `out/failed.txt` while keeping index alignment by writing a zero vector and a failure flag.

**Storage Plan**
- Preferred: per-folder `embeddings.npy` created via `numpy.lib.format.open_memmap`.
- Required metadata per folder:
  - `out/<folder_rel>/paths.txt` for index lookup.
  - `out/<folder_rel>/shape.json` including `num_images`, `embedding_dim`, `dtype`, and `model_id`.
- Size estimate formula:
  - Disk bytes ≈ `num_images * embedding_dim * dtype_bytes`.
  - If the file is too large or filesystem limits are a concern, switch to shards:
    - `embeddings_000.npy`, `embeddings_001.npy`, ...
    - `shards.json` with ranges and filenames.

**DINOv3-Specific Notes**
- Use float32 compute and store float32 embeddings.
- Pooling must exclude CLS and register tokens as done in `compare_embeddings_new_models.py`.

**GPU-Specific Notes (RTX4090)**
- Start with `batch_size=64` or `batch_size=96` and tune based on GPU memory usage.
- Enable TF32 as noted above for a large speedup with minimal numeric impact on embeddings.

**Future Proofing: SigLIP2 And big_city**
- Keep the model interface abstracted as `load_model_and_processor(model_key)` and `extract_embeddings(model_key, ...)`.
- Add a `--model` switch and store embeddings under `out/<model_key>/`.
- Support `--root MatrixCity/big_city` without changing the pipeline, only data size and output path.

**Validation**
- Verify `len(paths.txt)` equals `embeddings.npy.shape[0]`.
- Sample N images and recompute embeddings to ensure exact index mapping.
- Optionally run cosine similarity distribution sanity checks against a small subset.
