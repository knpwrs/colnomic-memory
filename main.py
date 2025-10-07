#!/usr/bin/env python3
"""
Script to generate embeddings for images using ColNomic Embed Multimodal 7B.
Supports multiple backends (CUDA, MPS, CPU) and reports peak memory usage.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from enum import Enum

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class Backend(Enum):
    """Supported compute backends."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

    @classmethod
    def detect_available(cls) -> "Backend":
        """Detect the best available backend."""
        if torch.cuda.is_available():
            return cls.CUDA
        elif torch.backends.mps.is_available():
            return cls.MPS
        else:
            return cls.CPU

    @classmethod
    def from_string(cls, backend_str: str) -> "Backend":
        """Convert string to Backend enum."""
        backend_map = {b.value: b for b in cls}
        if backend_str not in backend_map:
            raise ValueError(f"Unknown backend: {backend_str}. Choose from: {list(backend_map.keys())}")
        return backend_map[backend_str]


def load_images_from_directory(directory: Path) -> List[Image.Image]:
    """Load all image files from the specified directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        raise ValueError(f"No images found in directory: {directory}")

    print(f"Found {len(image_paths)} images in {directory}")
    images = []
    for path in sorted(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}", file=sys.stderr)

    return images


def get_memory_stats(backend: Backend) -> Tuple[float, float]:
    """
    Get peak memory usage in bytes and GB for the given backend.
    Returns (bytes, gigabytes).
    """
    if backend == Backend.CUDA:
        peak_bytes = torch.cuda.max_memory_allocated()
    elif backend == Backend.MPS:
        peak_bytes = torch.mps.driver_allocated_memory()
    else:  # CPU
        # For CPU, we can't easily track peak memory, return 0
        peak_bytes = 0

    peak_gb = peak_bytes / (1024 ** 3)
    return peak_bytes, peak_gb


def reset_memory_stats(backend: Backend) -> None:
    """Reset memory statistics for the given backend."""
    if backend == Backend.CUDA:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif backend == Backend.MPS:
        torch.mps.empty_cache()
    # CPU doesn't need cache clearing


def clear_cache(backend: Backend) -> None:
    """Clear memory cache for the given backend."""
    if backend == Backend.CUDA:
        torch.cuda.empty_cache()
    elif backend == Backend.MPS:
        torch.mps.empty_cache()


def process_images_in_batches(
    images: List[Image.Image],
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    batch_size: int,
    backend: Backend
) -> List[torch.Tensor]:
    """Process images in batches and generate embeddings."""
    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size} "
              f"({len(batch)} images)")

        batch_images = processor.process_images(batch).to(model.device)

        with torch.no_grad():
            embeddings = model(**batch_images)
            all_embeddings.append(embeddings.cpu())

        # Clear cache after each batch
        clear_cache(backend)

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for images using ColNomic Embed Multimodal 7B"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing images (default: 8)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nomic-ai/colnomic-embed-multimodal-7b",
        help="Model name/path (default: nomic-ai/colnomic-embed-multimodal-7b)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Compute backend to use (default: auto - auto-detect best available)"
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory does not exist: {args.directory}", file=sys.stderr)
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: Path is not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Determine backend
    if args.backend == "auto":
        backend = Backend.detect_available()
        print(f"Auto-detected backend: {backend.value}")
    else:
        backend = Backend.from_string(args.backend)
        # Validate that the requested backend is available
        if backend == Backend.CUDA and not torch.cuda.is_available():
            print("Error: CUDA backend requested but not available", file=sys.stderr)
            sys.exit(1)
        elif backend == Backend.MPS and not torch.backends.mps.is_available():
            print("Error: MPS backend requested but not available", file=sys.stderr)
            sys.exit(1)

    # Set device map based on backend
    if backend == Backend.CUDA:
        device_map = "cuda:0"
        dtype = torch.bfloat16
        use_flash_attn = is_flash_attn_2_available()
    elif backend == Backend.MPS:
        device_map = "mps"
        dtype = torch.float16  # MPS doesn't support bfloat16 well
        use_flash_attn = False  # Flash attention not available on MPS
    else:  # CPU
        device_map = "cpu"
        dtype = torch.float32  # CPU works best with float32
        use_flash_attn = False

    print(f"Using backend: {backend.value}")
    print(f"Using dtype: {dtype}")
    print(f"Loading model: {args.model_name}")

    # Load model
    model = ColQwen2_5.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attn else None,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)

    print("Model loaded successfully")

    # Load images
    images = load_images_from_directory(args.directory)

    # Reset peak memory stats
    reset_memory_stats(backend)

    # Process images
    print(f"\nProcessing {len(images)} images with batch size {args.batch_size}")
    embeddings = process_images_in_batches(images, model, processor, args.batch_size, backend)

    # Get peak memory usage
    peak_memory_bytes, peak_memory_gb = get_memory_stats(backend)
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Backend: {backend.value}")
    print(f"Total images processed: {len(images)}")
    print(f"Total batches: {len(embeddings)}")
    if backend != Backend.CPU:
        print(f"Peak {backend.value.upper()} memory used: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")
    else:
        print("Memory tracking not available for CPU backend")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
