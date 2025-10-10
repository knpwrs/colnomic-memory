# Nomic Memory

Generate image embeddings using Nomic's multimodal embedding models and DINOv2 vision models with support for multiple compute backends and memory usage tracking. Supports both ColNomic (multi-vector) and Nomic (single-vector) variants in 7B and 3B parameter sizes, as well as DINOv2 vision transformers.

## Features

- **Multiple model variants**: Support for ColNomic (multi-vector) and Nomic (single-vector) models in 7B and 3B sizes, plus DINOv2 vision models
- **Multi-backend support**: Automatically detects and uses the best available backend (CUDA, MPS, or CPU)
- **Memory tracking**: Reports peak memory usage during processing
- **Batch processing**: Configurable batch sizes for optimal performance
- **Multiple image formats**: Supports JPG, PNG, BMP, GIF, TIFF, and WebP
- **Backend-specific optimizations**:
  - **CUDA**: bfloat16 precision, Flash Attention 2 (if available for Nomic models)
  - **MPS**: float16 precision (Apple Silicon)
  - **CPU**: float32 precision

## Supported Models

### Nomic Multimodal Models

| Model Variant | Parameters | Type | Hugging Face Model | NDCG@5 (Vidore-v2) |
|---------------|------------|------|-------------------|-------------------|
| ColNomic 7B (default) | 7B | Multi-vector | `nomic-ai/colnomic-embed-multimodal-7b` | 62.7 |
| ColNomic 3B | 3B | Multi-vector | `nomic-ai/colnomic-embed-multimodal-3b` | 61.2 |
| Nomic 7B | 7B | Single-vector | `nomic-ai/nomic-embed-multimodal-7b` | 58.8 |
| Nomic 3B | 3B | Single-vector | `nomic-ai/nomic-embed-multimodal-3b` | 58.8 |

**ColNomic models** use multi-vector late interaction, creating multiple embeddings per document for more precise matching. Best for visual document retrieval tasks like research papers, technical docs, and product catalogs.

**Nomic models** use single-vector embeddings, offering a simpler dense embedding approach with lower storage requirements.

### DINOv2 Vision Models

| Model Variant | Parameters | Type | Hugging Face Model | Embedding Dim |
|---------------|------------|------|-------------------|---------------|
| DINOv2 Small | 22M | Vision Transformer | `facebook/dinov2-small` | 384 |
| DINOv2 Base | 86M | Vision Transformer | `facebook/dinov2-base` | 768 |
| DINOv2 Large | 300M | Vision Transformer | `facebook/dinov2-large` | 1024 |
| DINOv2 Giant | 1.1B | Vision Transformer | `facebook/dinov2-giant` | 1536 |

**DINOv2 models** are self-supervised vision transformers that produce high-quality image embeddings without any fine-tuning. They excel at capturing visual features for tasks like image classification, retrieval, and similarity search. Each image produces a single CLS token embedding representing the global image features.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. First, ensure you have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync dependencies:

```bash
uv sync
```

## Usage

### Basic Usage

Process all images in a directory with default settings (ColNomic 7B, auto-detected backend):

```bash
uv run python main.py /path/to/images
```

### Select Model Variant

Choose between different model variants:

```bash
# Nomic Models (Multimodal)
# Use ColNomic 7B (default - best performance for multimodal)
uv run python main.py /path/to/images --model-variant colnomic-7b

# Use ColNomic 3B (faster, lower memory)
uv run python main.py /path/to/images --model-variant colnomic-3b

# Use Nomic 7B (single-vector embeddings)
uv run python main.py /path/to/images --model-variant nomic-7b

# Use Nomic 3B (fastest, lowest memory)
uv run python main.py /path/to/images --model-variant nomic-3b

# DINOv2 Models (Vision-only)
# Use DINOv2 Small (lightweight, 22M parameters)
uv run python main.py /path/to/images --model-variant dinov2-small

# Use DINOv2 Base (balanced, 86M parameters)
uv run python main.py /path/to/images --model-variant dinov2-base

# Use DINOv2 Large (high quality, 300M parameters)
uv run python main.py /path/to/images --model-variant dinov2-large

# Use DINOv2 Giant (best quality, 1.1B parameters)
uv run python main.py /path/to/images --model-variant dinov2-giant
```

### Specify Backend

Force a specific compute backend:

```bash
# Use CUDA (NVIDIA GPUs)
uv run python main.py /path/to/images --backend cuda

# Use MPS (Apple Silicon)
uv run python main.py /path/to/images --backend mps

# Use CPU
uv run python main.py /path/to/images --backend cpu

# Auto-detect (default)
uv run python main.py /path/to/images --backend auto
```

### Configure Batch Size

Adjust batch size based on available memory:

```bash
# Smaller batch size for limited memory
uv run python main.py /path/to/images --batch-size 4

# Larger batch size for more memory
uv run python main.py /path/to/images --batch-size 16
```

### Use Custom Model

Specify a different model or local path (overrides --model-variant):

```bash
uv run python main.py /path/to/images --model-name /path/to/local/model
```

### Complete Examples

```bash
# Use Nomic multimodal model with CUDA
uv run python main.py ./my_images \
  --model-variant colnomic-3b \
  --backend cuda \
  --batch-size 8

# Use DINOv2 vision model with MPS (Apple Silicon)
uv run python main.py ./my_images \
  --model-variant dinov2-base \
  --backend mps \
  --batch-size 16

# Use DINOv2 Giant on CPU with smaller batch
uv run python main.py ./my_images \
  --model-variant dinov2-giant \
  --backend cpu \
  --batch-size 4
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | Path | Required | Directory containing images to process |
| `--model-variant` | str | colnomic-7b | Model variant: `colnomic-7b`, `colnomic-3b`, `nomic-7b`, `nomic-3b`, `dinov2-small`, `dinov2-base`, `dinov2-large`, or `dinov2-giant` |
| `--model-name` | str | None | Custom model name or path (overrides `--model-variant`) |
| `--batch-size` | int | 8 | Batch size for processing images |
| `--backend` | str | auto | Compute backend: `cuda`, `mps`, `cpu`, or `auto` |

## Output

The script provides detailed progress information and reports:

```
Using model variant: colnomic-7b (nomic-ai/colnomic-embed-multimodal-7b)
Auto-detected backend: cuda
Using backend: cuda
Using dtype: torch.bfloat16
Loading model: nomic-ai/colnomic-embed-multimodal-7b
Model loaded successfully
Found 100 images in /path/to/images

Processing 100 images with batch size 8
Processing batch 1/13 (8 images)
Processing batch 2/13 (8 images)
...

============================================================
Processing complete!
Backend: cuda
Total images processed: 100
Total batches: 13
Runtime: 120.45 seconds (2.01 minutes)
Peak CUDA memory used: 14.32 GB (14663.45 MB)
============================================================
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (for CUDA backend), Apple Silicon (for MPS backend), or CPU
- Dependencies managed via uv (see `pyproject.toml`)

## Testing

A test script is provided to verify the DINOv2 integration:

```bash
uv run python test_dinov2.py
```

The test script includes:

- Model type detection tests
- Model name mapping tests
- Optional model loading and embedding extraction test (requires downloading DINOv2 Small model)

## Benchmarking

A comprehensive benchmarking script is provided to test all model variants across different batch sizes:

```bash
uv run python benchmark_all.py
```

### Benchmark Features

- Tests all models (or a subset) with configurable batch sizes
- Records peak VRAM usage and processing time
- Saves results incrementally to markdown file
- Optional push notifications via [ntfy.sh](https://ntfy.sh)
- Handles errors gracefully with timeout protection (1 hour per run)

### Benchmark Usage

```bash
# Run all models with all batch sizes (default: 1, 4, 8, 16, 32, 64, 128)
uv run python benchmark_all.py

# Run only DINOv2 models
uv run python benchmark_all.py --models dinov2

# Run only Nomic models
uv run python benchmark_all.py --models nomic

# Run specific models with custom batch sizes
uv run python benchmark_all.py \
  --models dinov2-small dinov2-base \
  --batch-sizes 1 8 16

# Use specific backend and custom output file
uv run python benchmark_all.py \
  --backend cuda \
  --output my_benchmarks.md

# Run with push notifications (requires ntfy.sh topic)
uv run python benchmark_all.py \
  --ntfy-topic my-benchmark-notifications

# Full example with all options
uv run python benchmark_all.py \
  --models dinov2 \
  --batch-sizes 1 4 8 \
  --backend cuda \
  --output dinov2_benchmarks.md \
  --ntfy-topic gpu-benchmarks
```

### Benchmark Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image-dir` | Path | ./images | Directory containing images to process |
| `--backend` | str | auto | Compute backend: `cuda`, `mps`, `cpu`, or `auto` |
| `--output` | Path | benchmark_results.md | Output file for results |
| `--models` | str[] | all | Models to benchmark: specific variants, `nomic`, `dinov2`, or `all` |
| `--batch-sizes` | int[] | [1,4,8,16,32,64,128] | Batch sizes to test |
| `--ntfy-topic` | str | None | Optional ntfy.sh topic for push notifications |

### Push Notifications

The benchmark script can send push notifications via [ntfy.sh](https://ntfy.sh) to keep you updated on long-running benchmarks:

**Notifications sent for:**

- Benchmark suite start
- Each individual run completion (with runtime and VRAM stats)
- Each individual run failure/timeout/error
- Final suite completion summary

**To receive notifications:**

1. Choose a unique topic name (e.g., `my-gpu-benchmarks`)
2. Subscribe to your topic:
   - Web: Visit <https://ntfy.sh/your-topic-name>
   - Mobile: Download ntfy app and subscribe
   - Desktop: Use ntfy desktop client
3. Run benchmark with `--ntfy-topic your-topic-name`

Example:

```bash
# Start benchmark with notifications
uv run python benchmark_all.py \
  --models dinov2 \
  --ntfy-topic my-gpu-benchmarks

# Subscribe to receive updates at:
# https://ntfy.sh/my-gpu-benchmarks
```

### Benchmark Output

The script generates a markdown file with:

- Summary tables for each model variant
- Runtime, peak VRAM, and success/failure status
- Detailed JSON results at the end

Results are saved incrementally, so you can view progress even while benchmarks are running.

## Model Information

This script supports multiple embedding model families:

### Nomic Multimodal Models

#### ColNomic Models (Multi-vector)

- Use late interaction mechanism for more precise matching
- Create multiple embeddings per document/query
- Best performance on visual document retrieval benchmarks
- Ideal for: research papers, technical docs, product catalogs, financial reports
- Built on Qwen2.5-VL architecture

#### Nomic Models (Single-vector)

- Use dense single-vector embeddings
- Lower storage requirements
- Simpler retrieval pipeline
- Still achieve strong performance on multimodal tasks
- Built on Qwen2.5-VL architecture

All Nomic models:

- Excel at visual document retrieval
- Directly encode interleaved text and images
- Support multiple image formats

### DINOv2 Vision Models

- Self-supervised vision transformers trained on 142M images
- Produce high-quality visual features without fine-tuning
- Generate single CLS token embeddings (global image representation)
- Excellent for image classification, retrieval, and similarity search
- Ideal for: general image understanding, visual search, clustering
- Based on Vision Transformer (ViT) architecture
- No text encoding capability - purely visual features

### Choosing Between Model Families

**Use Nomic/ColNomic models when:**

- You need multimodal embeddings (text + images together)
- Working with visual documents (PDFs, slides, diagrams)
- Need to retrieve based on both visual and textual content
- Want state-of-the-art performance on document retrieval

**Use DINOv2 models when:**

- You only need pure image embeddings
- Working with photographs, artwork, or general images
- Want lighter-weight models with faster inference
- Need embeddings for image classification or clustering
- Want models that are more widely adopted in computer vision research

## Performance & Memory Usage

Benchmark results on various configurations (2000 images, NVIDIA H100 GPU):

### ColNomic 3B (Multi-vector)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Both encoders | 7.49 GB (7666.98 MB) | 10:16 | `uv run python main.py ./images --model-variant colnomic-3b --batch-size 1` |
| 4 | Both encoders | 9.44 GB (9661.52 MB) | 14:31 | `uv run python main.py ./images --model-variant colnomic-3b --batch-size 4` |
| 8 | Both encoders | 14.69 GB (15044.16 MB) | 20:52 | `uv run python main.py ./images --model-variant colnomic-3b --batch-size 8` |
| 16 | Both encoders | 34.15 GB (34964.95 MB) | 33:43 | `uv run python main.py ./images --model-variant colnomic-3b --batch-size 16` |
| 32 | Both encoders | OOM | Failed | `uv run python main.py ./images --model-variant colnomic-3b --batch-size 32` |

### ColNomic 7B (Multi-vector)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Both encoders | 14.92 GB (15280.17 MB) | 06:21 | `uv run python main.py ./images --model-variant colnomic-7b --batch-size 1` |
| 4 | Both encoders | 15.82 GB (16201.52 MB) | 07:38 | `uv run python main.py ./images --model-variant colnomic-7b --batch-size 4` |
| 8 | Both encoders | 17.69 GB (18113.69 MB) | 09:46 | `uv run python main.py ./images --model-variant colnomic-7b --batch-size 8` |
| 16 | Both encoders | 24.81 GB (25404.05 MB) | 14:11 | `uv run python main.py ./images --model-variant colnomic-7b --batch-size 16` |
| 32 | Both encoders | 51.42 GB (52652.81 MB) | 23:13 | `uv run python main.py ./images --model-variant colnomic-7b --batch-size 32` |

### Nomic 3B (Single-vector)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Both encoders | 7.35 GB (7528.37 MB) | 06:23 | `uv run python main.py ./images --model-variant nomic-3b --batch-size 1` |
| 4 | Both encoders | 8.15 GB (8341.92 MB) | 07:18 | `uv run python main.py ./images --model-variant nomic-3b --batch-size 4` |
| 8 | Both encoders | 10.18 GB (10428.73 MB) | 09:21 | `uv run python main.py ./images --model-variant nomic-3b --batch-size 8` |
| 16 | Both encoders | 17.29 GB (17701.72 MB) | 13:44 | `uv run python main.py ./images --model-variant nomic-3b --batch-size 16` |
| 32 | Both encoders | 43.86 GB (44914.38 MB) | 22:45 | `uv run python main.py ./images --model-variant nomic-3b --batch-size 32` |

### Nomic 7B (Single-vector)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Both encoders | 14.92 GB (15273.78 MB) | 06:26 | `uv run python main.py ./images --model-variant nomic-7b --batch-size 1` |
| 4 | Both encoders | 15.82 GB (16194.58 MB) | 07:41 | `uv run python main.py ./images --model-variant nomic-7b --batch-size 4` |
| 8 | Both encoders | 17.68 GB (18106.06 MB) | 09:48 | `uv run python main.py ./images --model-variant nomic-7b --batch-size 8` |
| 16 | Both encoders | 24.80 GB (25397.00 MB) | 14:08 | `uv run python main.py ./images --model-variant nomic-7b --batch-size 16` |
| 32 | Both encoders | 51.41 GB (52645.00 MB) | 23:10 | `uv run python main.py ./images --model-variant nomic-7b --batch-size 32` |

### DINOv2 Small (22M params)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Vision only | 0.08 GB (78.93 MB) | 00:29 | `uv run python main.py ./images --model-variant dinov2-small --batch-size 1` |
| 4 | Vision only | 0.09 GB (89.07 MB) | 00:23 | `uv run python main.py ./images --model-variant dinov2-small --batch-size 4` |
| 8 | Vision only | 0.10 GB (104.59 MB) | 00:21 | `uv run python main.py ./images --model-variant dinov2-small --batch-size 8` |
| 16 | Vision only | 0.13 GB (129.62 MB) | 00:21 | `uv run python main.py ./images --model-variant dinov2-small --batch-size 16` |
| 32 | Vision only | 0.18 GB (183.83 MB) | 00:21 | `uv run python main.py ./images --model-variant dinov2-small --batch-size 32` |

### DINOv2 Base (86M params)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Vision only | 0.20 GB (209.52 MB) | 00:29 | `uv run python main.py ./images --model-variant dinov2-base --batch-size 1` |
| 4 | Vision only | 0.23 GB (230.51 MB) | 00:23 | `uv run python main.py ./images --model-variant dinov2-base --batch-size 4` |
| 8 | Vision only | 0.25 GB (253.40 MB) | 00:22 | `uv run python main.py ./images --model-variant dinov2-base --batch-size 8` |
| 16 | Vision only | 0.30 GB (302.80 MB) | 00:21 | `uv run python main.py ./images --model-variant dinov2-base --batch-size 16` |
| 32 | Vision only | 0.39 GB (402.34 MB) | 00:21 | `uv run python main.py ./images --model-variant dinov2-base --batch-size 32` |

### DINOv2 Large (300M params)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Vision only | 0.61 GB (620.87 MB) | 00:36 | `uv run python main.py ./images --model-variant dinov2-large --batch-size 1` |
| 4 | Vision only | 0.63 GB (645.71 MB) | 00:25 | `uv run python main.py ./images --model-variant dinov2-large --batch-size 4` |
| 8 | Vision only | 0.66 GB (678.70 MB) | 00:23 | `uv run python main.py ./images --model-variant dinov2-large --batch-size 8` |
| 16 | Vision only | 0.72 GB (742.37 MB) | 00:22 | `uv run python main.py ./images --model-variant dinov2-large --batch-size 16` |
| 32 | Vision only | 0.85 GB (872.03 MB) | 00:21 | `uv run python main.py ./images --model-variant dinov2-large --batch-size 32` |

### DINOv2 Giant (1.1B params)

| Batch Size | Mode | Peak VRAM Usage | Processing Time | Command |
|------------|------|-----------------|-----------------|---------|
| 1 | Vision only | 2.16 GB (2214.30 MB) | 00:47 | `uv run python main.py ./images --model-variant dinov2-giant --batch-size 1` |
| 4 | Vision only | 2.21 GB (2258.06 MB) | 00:28 | `uv run python main.py ./images --model-variant dinov2-giant --batch-size 4` |
| 8 | Vision only | 2.26 GB (2311.63 MB) | 00:26 | `uv run python main.py ./images --model-variant dinov2-giant --batch-size 8` |
| 16 | Vision only | 2.37 GB (2422.44 MB) | 00:24 | `uv run python main.py ./images --model-variant dinov2-giant --batch-size 16` |
| 32 | Vision only | 2.58 GB (2644.46 MB) | 00:24 | `uv run python main.py ./images --model-variant dinov2-giant --batch-size 32` |

## Sample Images

This repository includes sample images in the `images/` directory, sourced from [yavuzceliker/sample-images](https://github.com/yavuzceliker/sample-images).

## License

See LICENSE file for details.
