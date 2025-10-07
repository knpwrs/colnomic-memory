#!/usr/bin/env python3
"""
Script to generate embeddings for images using ColNomic Embed Multimodal 7B.
Reports peak CUDA memory usage during processing.
"""
import argparse
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


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


def process_images_in_batches(
    images: List[Image.Image],
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    batch_size: int
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory does not exist: {args.directory}", file=sys.stderr)
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: Path is not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. This script requires a CUDA-capable GPU.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model_name}")

    # Load model
    model = ColQwen2_5.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(args.model_name)

    print("Model loaded successfully")

    # Load images
    images = load_images_from_directory(args.directory)

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Process images
    print(f"\nProcessing {len(images)} images with batch size {args.batch_size}")
    embeddings = process_images_in_batches(images, model, processor, args.batch_size)

    # Get peak memory usage
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {len(images)}")
    print(f"Total batches: {len(embeddings)}")
    print(f"Peak CUDA memory used: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
