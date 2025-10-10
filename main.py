#!/usr/bin/env python3
"""
Script to generate embeddings for images using Nomic/ColNomic Embed Multimodal models
and DINOv2 vision models.
Supports multiple backends (CUDA, MPS, CPU) and reports peak memory usage.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from enum import Enum

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import AutoImageProcessor, AutoModel
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ModelType(Enum):
    """Model architecture types."""
    NOMIC = "nomic"  # Nomic/ColNomic models using ColQwen2_5
    DINOV2 = "dinov2"  # DINOv2 vision models


class ModelVariant(Enum):
    """Supported model variants."""
    # Nomic models
    COLNOMIC_7B = "colnomic-7b"
    COLNOMIC_3B = "colnomic-3b"
    NOMIC_7B = "nomic-7b"
    NOMIC_3B = "nomic-3b"

    # DINOv2 models
    DINOV2_SMALL = "dinov2-small"
    DINOV2_BASE = "dinov2-base"
    DINOV2_LARGE = "dinov2-large"
    DINOV2_GIANT = "dinov2-giant"

    def get_model_name(self) -> str:
        """Get the Hugging Face model identifier."""
        model_map = {
            # Nomic models
            ModelVariant.COLNOMIC_7B: "nomic-ai/colnomic-embed-multimodal-7b",
            ModelVariant.COLNOMIC_3B: "nomic-ai/colnomic-embed-multimodal-3b",
            ModelVariant.NOMIC_7B: "nomic-ai/nomic-embed-multimodal-7b",
            ModelVariant.NOMIC_3B: "nomic-ai/nomic-embed-multimodal-3b",

            # DINOv2 models
            ModelVariant.DINOV2_SMALL: "facebook/dinov2-small",
            ModelVariant.DINOV2_BASE: "facebook/dinov2-base",
            ModelVariant.DINOV2_LARGE: "facebook/dinov2-large",
            ModelVariant.DINOV2_GIANT: "facebook/dinov2-giant",
        }
        return model_map[self]

    def get_model_type(self) -> ModelType:
        """Get the model architecture type."""
        nomic_variants = {
            ModelVariant.COLNOMIC_7B,
            ModelVariant.COLNOMIC_3B,
            ModelVariant.NOMIC_7B,
            ModelVariant.NOMIC_3B,
        }
        if self in nomic_variants:
            return ModelType.NOMIC
        else:
            return ModelType.DINOV2

    @classmethod
    def from_string(cls, variant_str: str) -> "ModelVariant":
        """Convert string to ModelVariant enum."""
        variant_map = {v.value: v for v in cls}
        if variant_str not in variant_map:
            raise ValueError(f"Unknown model variant: {variant_str}. Choose from: {list(variant_map.keys())}")
        return variant_map[variant_str]


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
    model: Union[ColQwen2_5, AutoModel],
    processor: Union[ColQwen2_5_Processor, AutoImageProcessor],
    batch_size: int,
    backend: Backend,
    model_type: ModelType
) -> List[torch.Tensor]:
    """Process images in batches and generate embeddings with pipelined execution."""
    all_embeddings = []
    total_batches = (len(images) + batch_size - 1) // batch_size

    # Use CUDA streams for pipelined execution (CUDA only)
    use_streams = backend == Backend.CUDA
    if use_streams:
        # Create separate streams for data transfer and computation
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()

    # Prepare first batch
    current_batch_idx = 0
    current_batch = images[0:batch_size]

    # Preprocess first batch on CPU
    if model_type == ModelType.NOMIC:
        current_inputs = processor.process_images(current_batch)
    else:  # DINOV2
        current_inputs = processor(images=current_batch, return_tensors="pt")

    with torch.no_grad():
        while current_batch_idx < total_batches:
            batch_num = current_batch_idx + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(current_batch)} images)")

            # Prepare next batch while current one is processing (if available)
            next_batch_idx = current_batch_idx + 1
            if next_batch_idx < total_batches:
                next_start = next_batch_idx * batch_size
                next_end = min(next_start + batch_size, len(images))
                next_batch = images[next_start:next_end]

                # Preprocess next batch on CPU while GPU works on current batch
                if model_type == ModelType.NOMIC:
                    next_inputs = processor.process_images(next_batch)
                else:  # DINOV2
                    next_inputs = processor(images=next_batch, return_tensors="pt")
            else:
                next_inputs = None

            # Transfer current batch to GPU and compute
            if use_streams:
                with torch.cuda.stream(compute_stream):
                    current_inputs_gpu = {k: v.to(model.device, non_blocking=True)
                                         for k, v in current_inputs.items()}

                    if model_type == ModelType.NOMIC:
                        embeddings = model(**current_inputs_gpu)
                    else:  # DINOV2
                        outputs = model(**current_inputs_gpu)
                        embeddings = outputs.last_hidden_state[:, 0, :]

                    # Transfer results back to CPU asynchronously
                    embeddings_cpu = embeddings.cpu()

                # Wait for computation to finish
                compute_stream.synchronize()
            else:
                # Non-CUDA backends: standard synchronous processing
                current_inputs_gpu = {k: v.to(model.device)
                                     for k, v in current_inputs.items()}

                if model_type == ModelType.NOMIC:
                    embeddings = model(**current_inputs_gpu)
                else:  # DINOV2
                    outputs = model(**current_inputs_gpu)
                    embeddings = outputs.last_hidden_state[:, 0, :]

                embeddings_cpu = embeddings.cpu()

            all_embeddings.append(embeddings_cpu)

            # Delete processed batch tensors to free GPU memory
            del current_inputs_gpu
            del embeddings
            if model_type == ModelType.DINOV2:
                del outputs

            # Selective cache clearing - only if we've accumulated enough batches
            # This reduces the overhead of cache clearing while preventing OOM
            if (batch_num % 4 == 0) or (batch_num == total_batches):
                clear_cache(backend)

            # Move to next batch
            current_batch = next_batch if next_inputs is not None else None
            current_inputs = next_inputs
            current_batch_idx = next_batch_idx

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for images using Nomic/ColNomic Embed Multimodal models"
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
        "--model-variant",
        type=str,
        choices=[
            "colnomic-7b", "colnomic-3b", "nomic-7b", "nomic-3b",
            "dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"
        ],
        default="colnomic-7b",
        help="Model variant to use (default: colnomic-7b). ColNomic models use multi-vector embeddings, "
             "Nomic models use single-vector embeddings, DINOv2 models use vision transformers for embeddings."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name/path (overrides --model-variant if specified)"
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

    # Determine model to use
    if args.model_name:
        model_name = args.model_name
        model_type = None  # Will need to be inferred or specified separately
        print(f"Using custom model: {model_name}")
        print("Warning: Custom model path provided. Assuming Nomic model type.", file=sys.stderr)
        print("If using DINOv2, please use --model-variant instead.", file=sys.stderr)
        model_type = ModelType.NOMIC
    else:
        model_variant = ModelVariant.from_string(args.model_variant)
        model_name = model_variant.get_model_name()
        model_type = model_variant.get_model_type()
        print(f"Using model variant: {args.model_variant} ({model_name})")
        print(f"Model type: {model_type.value}")

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

    print(f"Loading model: {model_name}")

    # Load model and processor based on model type
    if model_type == ModelType.NOMIC:
        # Load Nomic/ColNomic model
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2" if use_flash_attn else None,
        ).eval()

        processor = ColQwen2_5_Processor.from_pretrained(model_name)

    elif model_type == ModelType.DINOV2:
        # Load DINOv2 model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device_map).eval()

        processor = AutoImageProcessor.from_pretrained(model_name)

    print("Model loaded successfully")

    # Load images
    images = load_images_from_directory(args.directory)

    # Reset peak memory stats
    reset_memory_stats(backend)

    # Process images
    print(f"\nProcessing {len(images)} images with batch size {args.batch_size}")
    start_time = time.time()
    embeddings = process_images_in_batches(
        images, model, processor, args.batch_size, backend, model_type
    )
    end_time = time.time()

    # Calculate runtime
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60

    # Get peak memory usage
    peak_memory_bytes, peak_memory_gb = get_memory_stats(backend)
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Backend: {backend.value}")
    print(f"Total images processed: {len(images)}")
    print(f"Total batches: {len(embeddings)}")
    print(f"Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")
    if backend != Backend.CPU:
        print(f"Peak {backend.value.upper()} memory used: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")
    else:
        print("Memory tracking not available for CPU backend")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
