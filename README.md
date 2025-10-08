# Nomic Memory

Generate image embeddings using Nomic's multimodal embedding models with support for multiple compute backends and memory usage tracking. Supports both ColNomic (multi-vector) and Nomic (single-vector) variants in 7B and 3B parameter sizes.

## Features

- **Multiple model variants**: Support for ColNomic (multi-vector) and Nomic (single-vector) models in 7B and 3B sizes
- **Multi-backend support**: Automatically detects and uses the best available backend (CUDA, MPS, or CPU)
- **Memory tracking**: Reports peak memory usage during processing
- **Batch processing**: Configurable batch sizes for optimal performance
- **Multiple image formats**: Supports JPG, PNG, BMP, GIF, TIFF, and WebP
- **Backend-specific optimizations**:
  - **CUDA**: bfloat16 precision, Flash Attention 2 (if available)
  - **MPS**: float16 precision (Apple Silicon)
  - **CPU**: float32 precision

## Supported Models

| Model Variant | Parameters | Type | Hugging Face Model | NDCG@5 (Vidore-v2) |
|---------------|------------|------|-------------------|-------------------|
| ColNomic 7B (default) | 7B | Multi-vector | `nomic-ai/colnomic-embed-multimodal-7b` | 62.7 |
| ColNomic 3B | 3B | Multi-vector | `nomic-ai/colnomic-embed-multimodal-3b` | 61.2 |
| Nomic 7B | 7B | Single-vector | `nomic-ai/nomic-embed-multimodal-7b` | 58.8 |
| Nomic 3B | 3B | Single-vector | `nomic-ai/nomic-embed-multimodal-3b` | 58.8 |

**ColNomic models** use multi-vector late interaction, creating multiple embeddings per document for more precise matching. Best for visual document retrieval tasks like research papers, technical docs, and product catalogs.

**Nomic models** use single-vector embeddings, offering a simpler dense embedding approach with lower storage requirements.

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
# Use ColNomic 7B (default - best performance)
uv run python main.py /path/to/images --model-variant colnomic-7b

# Use ColNomic 3B (faster, lower memory)
uv run python main.py /path/to/images --model-variant colnomic-3b

# Use Nomic 7B (single-vector embeddings)
uv run python main.py /path/to/images --model-variant nomic-7b

# Use Nomic 3B (fastest, lowest memory)
uv run python main.py /path/to/images --model-variant nomic-3b
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

### Complete Example

```bash
uv run python main.py ./my_images \
  --model-variant colnomic-3b \
  --backend cuda \
  --batch-size 8
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | Path | Required | Directory containing images to process |
| `--model-variant` | str | colnomic-7b | Model variant: `colnomic-7b`, `colnomic-3b`, `nomic-7b`, or `nomic-3b` |
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
Peak CUDA memory used: 14.32 GB (14663.45 MB)
============================================================
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (for CUDA backend), Apple Silicon (for MPS backend), or CPU
- Dependencies managed via uv (see `pyproject.toml`)

## Model Information

This script supports four Nomic multimodal embedding models:

### ColNomic Models (Multi-vector)

- Use late interaction mechanism for more precise matching
- Create multiple embeddings per document/query
- Best performance on visual document retrieval benchmarks
- Ideal for: research papers, technical docs, product catalogs, financial reports

### Nomic Models (Single-vector)

- Use dense single-vector embeddings
- Lower storage requirements
- Simpler retrieval pipeline
- Still achieve strong performance on multimodal tasks

All models:

- Excel at visual document retrieval
- Directly encode interleaved text and images
- Support multiple image formats
- Built on Qwen2.5-VL architecture

## Sample Images

This repository includes sample images in the `images/` directory, sourced from [yavuzceliker/sample-images](https://github.com/yavuzceliker/sample-images).

## License

See LICENSE file for details.
