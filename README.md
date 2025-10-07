# ColNomic Memory

Generate image embeddings using the [ColNomic Embed Multimodal 7B](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) model with support for multiple compute backends and memory usage tracking.

## Features

- **Multi-backend support**: Automatically detects and uses the best available backend (CUDA, MPS, or CPU)
- **Memory tracking**: Reports peak memory usage during processing
- **Batch processing**: Configurable batch sizes for optimal performance
- **Multiple image formats**: Supports JPG, PNG, BMP, GIF, TIFF, and WebP
- **Backend-specific optimizations**:
  - **CUDA**: bfloat16 precision, Flash Attention 2 (if available)
  - **MPS**: float16 precision (Apple Silicon)
  - **CPU**: float32 precision

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

Process all images in a directory with auto-detected backend:

```bash
uv run python main.py /path/to/images
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

Specify a different model or local path:

```bash
uv run python main.py /path/to/images --model-name /path/to/local/model
```

### Complete Example

```bash
uv run python main.py ./my_images \
  --backend cuda \
  --batch-size 8 \
  --model-name nomic-ai/colnomic-embed-multimodal-7b
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | Path | Required | Directory containing images to process |
| `--batch-size` | int | 8 | Batch size for processing images |
| `--backend` | str | auto | Compute backend: `cuda`, `mps`, `cpu`, or `auto` |
| `--model-name` | str | nomic-ai/colnomic-embed-multimodal-7b | Model name or path |

## Output

The script provides detailed progress information and reports:

```
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
Peak CUDA memory used: 14.32 GB (14663.45 MB)
============================================================
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (for CUDA backend), Apple Silicon (for MPS backend), or CPU
- Dependencies managed via uv (see `pyproject.toml`)

## Model Information

This script uses the [ColNomic Embed Multimodal 7B](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) model, which:

- Excels at visual document retrieval
- Directly encodes interleaved text and images
- Achieves 62.7 NDCG@5 on Vidore-v2 benchmark
- Best for research papers, technical docs, product catalogs, and financial reports

## License

See LICENSE file for details.
