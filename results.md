# Benchmark Results

Generated: 2025-10-09 08:20:56

## COLNOMIC-3B

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 10.27 min | 7.49 GB | 2000 | 2000 |
| 4 | success | 14.52 min | 9.44 GB | 2000 | 500 |
| 8 | success | 20.86 min | 14.69 GB | 2000 | 250 |
| 16 | success | 33.71 min | 34.15 GB | 2000 | 125 |
| 32 | failed | Failed | N/A | N/A | N/A |

## COLNOMIC-7B

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 6.35 min | 14.92 GB | 2000 | 2000 |
| 4 | success | 7.64 min | 15.82 GB | 2000 | 500 |
| 8 | success | 9.77 min | 17.69 GB | 2000 | 250 |
| 16 | success | 14.19 min | 24.81 GB | 2000 | 125 |
| 32 | success | 23.21 min | 51.42 GB | 2000 | 63 |

## DINOV2-BASE

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 0.49 min | 0.20 GB | 2000 | 2000 |
| 4 | success | 0.38 min | 0.23 GB | 2000 | 500 |
| 8 | success | 0.36 min | 0.25 GB | 2000 | 250 |
| 16 | success | 0.35 min | 0.30 GB | 2000 | 125 |
| 32 | success | 0.34 min | 0.39 GB | 2000 | 63 |

## DINOV2-GIANT

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 0.78 min | 2.16 GB | 2000 | 2000 |
| 4 | success | 0.47 min | 2.21 GB | 2000 | 500 |
| 8 | success | 0.43 min | 2.26 GB | 2000 | 250 |
| 16 | success | 0.40 min | 2.37 GB | 2000 | 125 |
| 32 | success | 0.40 min | 2.58 GB | 2000 | 63 |

## DINOV2-LARGE

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 0.60 min | 0.61 GB | 2000 | 2000 |
| 4 | success | 0.41 min | 0.63 GB | 2000 | 500 |
| 8 | success | 0.38 min | 0.66 GB | 2000 | 250 |
| 16 | success | 0.36 min | 0.72 GB | 2000 | 125 |
| 32 | success | 0.36 min | 0.85 GB | 2000 | 63 |

## DINOV2-SMALL

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 0.48 min | 0.08 GB | 2000 | 2000 |
| 4 | success | 0.38 min | 0.09 GB | 2000 | 500 |
| 8 | success | 0.36 min | 0.10 GB | 2000 | 250 |
| 16 | success | 0.35 min | 0.13 GB | 2000 | 125 |
| 32 | success | 0.35 min | 0.18 GB | 2000 | 63 |

## NOMIC-3B

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 6.38 min | 7.35 GB | 2000 | 2000 |
| 4 | success | 7.31 min | 8.15 GB | 2000 | 500 |
| 8 | success | 9.35 min | 10.18 GB | 2000 | 250 |
| 16 | success | 13.73 min | 17.29 GB | 2000 | 125 |
| 32 | success | 22.75 min | 43.86 GB | 2000 | 63 |

## NOMIC-7B

| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |
|------------|--------|---------|-----------|--------|---------|
| 1 | success | 6.44 min | 14.92 GB | 2000 | 2000 |
| 4 | success | 7.68 min | 15.82 GB | 2000 | 500 |
| 8 | success | 9.80 min | 17.68 GB | 2000 | 250 |
| 16 | success | 14.13 min | 24.80 GB | 2000 | 125 |
| 32 | success | 23.16 min | 51.41 GB | 2000 | 63 |

## Detailed Results (JSON)

```json
[
  {
    "model_variant": "colnomic-7b",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 408.13266611099243,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 380.75,
    "runtime_minutes": 6.35,
    "peak_memory_gb": 14.92,
    "peak_memory_mb": 15280.17
  },
  {
    "model_variant": "colnomic-7b",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 485.5718548297882,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 458.33,
    "runtime_minutes": 7.64,
    "peak_memory_gb": 15.82,
    "peak_memory_mb": 16201.52
  },
  {
    "model_variant": "colnomic-7b",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 613.4669740200043,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 586.07,
    "runtime_minutes": 9.77,
    "peak_memory_gb": 17.69,
    "peak_memory_mb": 18113.69
  },
  {
    "model_variant": "colnomic-7b",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 878.4300236701965,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 851.15,
    "runtime_minutes": 14.19,
    "peak_memory_gb": 24.81,
    "peak_memory_mb": 25404.05
  },
  {
    "model_variant": "colnomic-7b",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 1420.1419167518616,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 1392.83,
    "runtime_minutes": 23.21,
    "peak_memory_gb": 51.42,
    "peak_memory_mb": 52652.81
  },
  {
    "model_variant": "colnomic-3b",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 651.0720958709717,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 615.93,
    "runtime_minutes": 10.27,
    "peak_memory_gb": 7.49,
    "peak_memory_mb": 7666.98
  },
  {
    "model_variant": "colnomic-3b",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 895.6514830589294,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 871.49,
    "runtime_minutes": 14.52,
    "peak_memory_gb": 9.44,
    "peak_memory_mb": 9661.52
  },
  {
    "model_variant": "colnomic-3b",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 1275.794763803482,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 1251.78,
    "runtime_minutes": 20.86,
    "peak_memory_gb": 14.69,
    "peak_memory_mb": 15044.16
  },
  {
    "model_variant": "colnomic-3b",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 2045.7341856956482,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 2022.64,
    "runtime_minutes": 33.71,
    "peak_memory_gb": 34.15,
    "peak_memory_mb": 34964.95
  },
  {
    "model_variant": "colnomic-3b",
    "batch_size": 32,
    "status": "failed",
    "error": "\nFetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]\nFetching 2 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 2/2 [00:00<00:00, 41734.37it/s]\n\nLoading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]\nLoading checkpoint shards:  50%|\u2588\u2588\u2588\u2588\u2588     | 1/2 [00:00<00:00,  1.08it/s]\nLoading checkpoint shards: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 2/2 [00:01<00:00,  1.59it/s]\nLoading checkpoint shards: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 2/2 [00:01<00:00,  1.48it/s]\nUsing a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.\nYou have video processor config saved in `preprocessor.json` file which is deprecated. Video processor configs should be saved in their own `video_preprocessor.json` file. You can rename the file or load and save the processor back which renames it automatically. Loading from `preprocessor.json` will be removed in v5.0.\nTraceback (most recent call last):\n  File \"/home/ubuntu/nomic-memory/main.py\", line 360, in <module>\n    main()\n  File \"/home/ubuntu/nomic-memory/main.py\", line 333, in main\n    embeddings = process_images_in_batches(\n                 ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/main.py\", line 183, in process_images_in_batches\n    embeddings = model(**batch_images)\n                 ^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1751, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1762, in _call_impl\n    return forward_call(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/colpali_engine/models/qwen2_5/colqwen2_5/modeling_colqwen2_5.py\", line 50, in forward\n    super()\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py\", line 1250, in forward\n    image_embeds = self.get_image_features(pixel_values, image_grid_thw)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py\", line 1200, in get_image_features\n    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1751, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1762, in _call_impl\n    return forward_call(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py\", line 492, in forward\n    hidden_states = blk(\n                    ^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py\", line 83, in __call__\n    return super().__call__(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1751, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1762, in _call_impl\n    return forward_call(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py\", line 286, in forward\n    hidden_states = hidden_states + self.attn(\n                                    ^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1751, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py\", line 1762, in _call_impl\n    return forward_call(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py\", line 248, in forward\n    attn_output, _ = attention_interface(\n                     ^^^^^^^^^^^^^^^^^^^^\n  File \"/home/ubuntu/nomic-memory/.venv/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py\", line 66, in sdpa_attention_forward\n    attn_output = torch.nn.functional.scaled_dot_product_attention(\n                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 47.36 GiB. GPU 0 has a total capacity of 79.18 GiB of which 17.85 GiB is free. Including non-PyTorch memory, this process has 61.32 GiB memory in use. Of the allocated memory 60.39 GiB is allocated by PyTorch, and 268.86 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)\n",
    "wall_time_seconds": 25.574488401412964
  },
  {
    "model_variant": "nomic-7b",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 433.45156025886536,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 386.64,
    "runtime_minutes": 6.44,
    "peak_memory_gb": 14.92,
    "peak_memory_mb": 15273.78
  },
  {
    "model_variant": "nomic-7b",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 486.1631145477295,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 460.83,
    "runtime_minutes": 7.68,
    "peak_memory_gb": 15.82,
    "peak_memory_mb": 16194.58
  },
  {
    "model_variant": "nomic-7b",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 612.0195510387421,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 588.16,
    "runtime_minutes": 9.8,
    "peak_memory_gb": 17.68,
    "peak_memory_mb": 18106.06
  },
  {
    "model_variant": "nomic-7b",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 873.2258303165436,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 848.03,
    "runtime_minutes": 14.13,
    "peak_memory_gb": 24.8,
    "peak_memory_mb": 25397.0
  },
  {
    "model_variant": "nomic-7b",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 1414.3532679080963,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 1389.67,
    "runtime_minutes": 23.16,
    "peak_memory_gb": 51.41,
    "peak_memory_mb": 52645.0
  },
  {
    "model_variant": "nomic-3b",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 416.43411779403687,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 382.72,
    "runtime_minutes": 6.38,
    "peak_memory_gb": 7.35,
    "peak_memory_mb": 7528.37
  },
  {
    "model_variant": "nomic-3b",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 462.4084310531616,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 438.42,
    "runtime_minutes": 7.31,
    "peak_memory_gb": 8.15,
    "peak_memory_mb": 8341.92
  },
  {
    "model_variant": "nomic-3b",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 584.6529638767242,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 561.07,
    "runtime_minutes": 9.35,
    "peak_memory_gb": 10.18,
    "peak_memory_mb": 10428.73
  },
  {
    "model_variant": "nomic-3b",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 846.7972991466522,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 823.81,
    "runtime_minutes": 13.73,
    "peak_memory_gb": 17.29,
    "peak_memory_mb": 17701.72
  },
  {
    "model_variant": "nomic-3b",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 1387.9454667568207,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 1365.15,
    "runtime_minutes": 22.75,
    "peak_memory_gb": 43.86,
    "peak_memory_mb": 44914.38
  },
  {
    "model_variant": "dinov2-small",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 49.675240993499756,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 29.03,
    "runtime_minutes": 0.48,
    "peak_memory_gb": 0.08,
    "peak_memory_mb": 78.93
  },
  {
    "model_variant": "dinov2-small",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 42.14215540885925,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 22.92,
    "runtime_minutes": 0.38,
    "peak_memory_gb": 0.09,
    "peak_memory_mb": 89.07
  },
  {
    "model_variant": "dinov2-small",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 41.06352710723877,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 21.42,
    "runtime_minutes": 0.36,
    "peak_memory_gb": 0.1,
    "peak_memory_mb": 104.59
  },
  {
    "model_variant": "dinov2-small",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 40.30585193634033,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 20.9,
    "runtime_minutes": 0.35,
    "peak_memory_gb": 0.13,
    "peak_memory_mb": 129.62
  },
  {
    "model_variant": "dinov2-small",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 40.37480545043945,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 20.99,
    "runtime_minutes": 0.35,
    "peak_memory_gb": 0.18,
    "peak_memory_mb": 183.83
  },
  {
    "model_variant": "dinov2-base",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 50.685598850250244,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 29.48,
    "runtime_minutes": 0.49,
    "peak_memory_gb": 0.2,
    "peak_memory_mb": 209.52
  },
  {
    "model_variant": "dinov2-base",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 42.13664364814758,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 22.75,
    "runtime_minutes": 0.38,
    "peak_memory_gb": 0.23,
    "peak_memory_mb": 230.51
  },
  {
    "model_variant": "dinov2-base",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 41.24011278152466,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 21.7,
    "runtime_minutes": 0.36,
    "peak_memory_gb": 0.25,
    "peak_memory_mb": 253.4
  },
  {
    "model_variant": "dinov2-base",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 40.307361125946045,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 20.9,
    "runtime_minutes": 0.35,
    "peak_memory_gb": 0.3,
    "peak_memory_mb": 302.8
  },
  {
    "model_variant": "dinov2-base",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 39.77614951133728,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 20.57,
    "runtime_minutes": 0.34,
    "peak_memory_gb": 0.39,
    "peak_memory_mb": 402.34
  },
  {
    "model_variant": "dinov2-large",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 58.057437896728516,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 35.81,
    "runtime_minutes": 0.6,
    "peak_memory_gb": 0.61,
    "peak_memory_mb": 620.87
  },
  {
    "model_variant": "dinov2-large",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 44.34682059288025,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 24.54,
    "runtime_minutes": 0.41,
    "peak_memory_gb": 0.63,
    "peak_memory_mb": 645.71
  },
  {
    "model_variant": "dinov2-large",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 42.326220989227295,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 22.61,
    "runtime_minutes": 0.38,
    "peak_memory_gb": 0.66,
    "peak_memory_mb": 678.7
  },
  {
    "model_variant": "dinov2-large",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 40.75389742851257,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 21.75,
    "runtime_minutes": 0.36,
    "peak_memory_gb": 0.72,
    "peak_memory_mb": 742.37
  },
  {
    "model_variant": "dinov2-large",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 40.499643087387085,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 21.41,
    "runtime_minutes": 0.36,
    "peak_memory_gb": 0.85,
    "peak_memory_mb": 872.03
  },
  {
    "model_variant": "dinov2-giant",
    "batch_size": 1,
    "status": "success",
    "wall_time_seconds": 70.65683460235596,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 2000,
    "runtime_seconds": 46.55,
    "runtime_minutes": 0.78,
    "peak_memory_gb": 2.16,
    "peak_memory_mb": 2214.3
  },
  {
    "model_variant": "dinov2-giant",
    "batch_size": 4,
    "status": "success",
    "wall_time_seconds": 48.06338286399841,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 500,
    "runtime_seconds": 27.97,
    "runtime_minutes": 0.47,
    "peak_memory_gb": 2.21,
    "peak_memory_mb": 2258.06
  },
  {
    "model_variant": "dinov2-giant",
    "batch_size": 8,
    "status": "success",
    "wall_time_seconds": 45.38658118247986,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 250,
    "runtime_seconds": 25.71,
    "runtime_minutes": 0.43,
    "peak_memory_gb": 2.26,
    "peak_memory_mb": 2311.63
  },
  {
    "model_variant": "dinov2-giant",
    "batch_size": 16,
    "status": "success",
    "wall_time_seconds": 43.6730580329895,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 125,
    "runtime_seconds": 24.18,
    "runtime_minutes": 0.4,
    "peak_memory_gb": 2.37,
    "peak_memory_mb": 2422.44
  },
  {
    "model_variant": "dinov2-giant",
    "batch_size": 32,
    "status": "success",
    "wall_time_seconds": 43.45715141296387,
    "backend": "cuda",
    "total_images": 2000,
    "total_batches": 63,
    "runtime_seconds": 23.79,
    "runtime_minutes": 0.4,
    "peak_memory_gb": 2.58,
    "peak_memory_mb": 2644.46
  }
]
```