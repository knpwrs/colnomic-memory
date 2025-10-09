#!/usr/bin/env python3
"""
Benchmark script to test all model variants with different batch sizes.
Records peak memory usage and processing time for each configuration.
Supports ntfy.sh notifications for progress updates.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import requests


# Model configurations
NOMIC_MODELS = [
    "colnomic-7b",
    "colnomic-3b",
    "nomic-7b",
    "nomic-3b",
]

DINOV2_MODELS = [
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
]

BATCH_SIZES = [1, 4, 8, 16, 32]


def send_ntfy_notification(
    topic: str,
    message: str,
    title: Optional[str] = None,
    priority: str = "default",
    tags: Optional[List[str]] = None
) -> bool:
    """
    Send a notification via ntfy.sh.

    Args:
        topic: The ntfy topic to send to
        message: The notification message
        title: Optional title for the notification
        priority: Priority level (min, low, default, high, urgent)
        tags: Optional list of tags/emojis

    Returns:
        True if successful, False otherwise
    """
    try:
        url = f"https://ntfy.sh/{topic}"
        headers = {}

        if title:
            headers["Title"] = title
        headers["Priority"] = priority
        if tags:
            headers["Tags"] = ",".join(tags)

        response = requests.post(url, data=message.encode('utf-8'), headers=headers)
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Failed to send ntfy notification: {e}", file=sys.stderr)
        return False


def parse_output(output: str) -> Dict[str, Any]:
    """Parse the output from main.py to extract metrics."""
    result = {
        "backend": None,
        "total_images": None,
        "total_batches": None,
        "runtime_seconds": None,
        "runtime_minutes": None,
        "peak_memory_gb": None,
        "peak_memory_mb": None,
    }

    for line in output.split('\n'):
        if "Backend:" in line and "Auto-detected" not in line and "Using backend:" not in line:
            result["backend"] = line.split("Backend:")[-1].strip()
        elif "Total images processed:" in line:
            result["total_images"] = int(line.split(":")[-1].strip())
        elif "Total batches:" in line:
            result["total_batches"] = int(line.split(":")[-1].strip())
        elif "Runtime:" in line:
            # Format: "Runtime: X.XX seconds (Y.YY minutes)"
            parts = line.split("Runtime:")[-1].strip()
            seconds = float(parts.split("seconds")[0].strip())
            minutes = float(parts.split("(")[1].split("minutes")[0].strip())
            result["runtime_seconds"] = seconds
            result["runtime_minutes"] = minutes
        elif "Peak" in line and "memory used:" in line:
            # Format: "Peak CUDA memory used: X.XX GB (Y.YY MB)"
            parts = line.split(":")[-1].strip()
            gb = float(parts.split("GB")[0].strip())
            mb = float(parts.split("(")[1].split("MB")[0].strip())
            result["peak_memory_gb"] = gb
            result["peak_memory_mb"] = mb

    return result


def run_benchmark(
    model_variant: str,
    batch_size: int,
    image_dir: Path,
    backend: str = "auto",
    image_only: bool = False,
    ntfy_topic: Optional[str] = None
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    cmd = [
        "uv", "run", "python", "main.py",
        str(image_dir),
        "--model-variant", model_variant,
        "--batch-size", str(batch_size),
        "--backend", backend,
    ]

    if image_only:
        cmd.append("--image-only")

    mode = "image-only" if image_only else "full"
    print(f"\n{'='*70}")
    print(f"Running: {model_variant} | batch={batch_size} | mode={mode}")
    print(f"{'='*70}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        end_time = time.time()

        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")

            # Send failure notification
            if ntfy_topic:
                send_ntfy_notification(
                    ntfy_topic,
                    f"Failed: {model_variant} batch={batch_size} mode={mode}\nError: {result.stderr[:200]}",
                    title="Benchmark Failed",
                    priority="high",
                    tags=["x", "warning"]
                )

            return {
                "model_variant": model_variant,
                "batch_size": batch_size,
                "image_only": image_only,
                "mode": mode,
                "status": "failed",
                "error": result.stderr,
                "wall_time_seconds": end_time - start_time,
            }

        # Parse the output
        metrics = parse_output(result.stdout)

        # Send success notification
        if ntfy_topic:
            runtime_str = f"{metrics.get('runtime_minutes', 0):.2f} min"
            memory_str = f"{metrics.get('peak_memory_gb', 0):.2f} GB" if metrics.get('peak_memory_gb') else "N/A"
            send_ntfy_notification(
                ntfy_topic,
                f"Completed: {model_variant} batch={batch_size} mode={mode}\nRuntime: {runtime_str}\nVRAM: {memory_str}",
                title="Benchmark Complete",
                priority="default",
                tags=["white_check_mark"]
            )

        return {
            "model_variant": model_variant,
            "batch_size": batch_size,
            "image_only": image_only,
            "mode": mode,
            "status": "success",
            "wall_time_seconds": end_time - start_time,
            **metrics
        }

    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after 1 hour")

        # Send timeout notification
        if ntfy_topic:
            send_ntfy_notification(
                ntfy_topic,
                f"Timeout: {model_variant} batch={batch_size} mode={mode}\nExceeded 1 hour limit",
                title="Benchmark Timeout",
                priority="high",
                tags=["alarm_clock", "warning"]
            )

        return {
            "model_variant": model_variant,
            "batch_size": batch_size,
            "image_only": image_only,
            "mode": mode,
            "status": "timeout",
            "error": "Command timed out after 1 hour",
        }
    except Exception as e:
        print(f"ERROR: {e}")

        # Send error notification
        if ntfy_topic:
            send_ntfy_notification(
                ntfy_topic,
                f"Error: {model_variant} batch={batch_size} mode={mode}\n{str(e)}",
                title="Benchmark Error",
                priority="high",
                tags=["x", "warning"]
            )

        return {
            "model_variant": model_variant,
            "batch_size": batch_size,
            "image_only": image_only,
            "mode": mode,
            "status": "error",
            "error": str(e),
        }


def format_results_as_markdown(results: List[Dict[str, Any]]) -> str:
    """Format benchmark results as markdown tables."""
    output = ["# Benchmark Results", ""]
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")

    # Group by model variant
    models = {}
    for result in results:
        model = result["model_variant"]
        if model not in models:
            models[model] = []
        models[model].append(result)

    # Create tables for each model
    for model, model_results in sorted(models.items()):
        output.append(f"## {model.upper()}")
        output.append("")

        # Separate by mode
        modes = {}
        for result in model_results:
            mode = result["mode"]
            if mode not in modes:
                modes[mode] = []
            modes[mode].append(result)

        for mode, mode_results in sorted(modes.items()):
            if len(modes) > 1:  # Only show mode header if there are multiple modes
                output.append(f"### Mode: {mode}")
                output.append("")

            # Create table header
            output.append("| Batch Size | Status | Runtime | Peak VRAM | Images | Batches |")
            output.append("|------------|--------|---------|-----------|--------|---------|")

            # Sort by batch size
            for result in sorted(mode_results, key=lambda x: x["batch_size"]):
                batch_size = result["batch_size"]
                status = result["status"]

                if status == "success":
                    runtime_str = f"{result['runtime_minutes']:.2f} min"
                    if result['peak_memory_gb'] is not None:
                        memory_str = f"{result['peak_memory_gb']:.2f} GB"
                    else:
                        memory_str = "N/A"
                    images_str = str(result.get('total_images', 'N/A'))
                    batches_str = str(result.get('total_batches', 'N/A'))
                else:
                    runtime_str = "Failed"
                    memory_str = "N/A"
                    images_str = "N/A"
                    batches_str = "N/A"

                output.append(f"| {batch_size} | {status} | {runtime_str} | {memory_str} | {images_str} | {batches_str} |")

            output.append("")

    # Add detailed results section
    output.append("## Detailed Results (JSON)")
    output.append("")
    output.append("```json")
    output.append(json.dumps(results, indent=2))
    output.append("```")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for all model variants and batch sizes"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("./images"),
        help="Directory containing images to process (default: ./images)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["cuda", "mps", "cpu", "auto"],
        help="Compute backend to use (default: auto)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.md"),
        help="Output file for results (default: benchmark_results.md)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=NOMIC_MODELS + DINOV2_MODELS + ["nomic", "dinov2", "all"],
        default=["all"],
        help="Models to benchmark (default: all)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=BATCH_SIZES,
        help=f"Batch sizes to test (default: {BATCH_SIZES})"
    )
    parser.add_argument(
        "--skip-nomic-full",
        action="store_true",
        help="Skip full mode for Nomic models (only run image-only mode)"
    )
    parser.add_argument(
        "--ntfy-topic",
        type=str,
        default=None,
        help="Optional ntfy.sh topic for notifications on success/failure"
    )

    args = parser.parse_args()

    # Validate image directory
    if not args.image_dir.exists():
        print(f"Error: Image directory does not exist: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.image_dir.is_dir():
        print(f"Error: Path is not a directory: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine which models to run
    models_to_run = []
    if "all" in args.models:
        models_to_run = NOMIC_MODELS + DINOV2_MODELS
    else:
        if "nomic" in args.models:
            models_to_run.extend(NOMIC_MODELS)
        if "dinov2" in args.models:
            models_to_run.extend(DINOV2_MODELS)
        # Add individual models
        for model in args.models:
            if model in NOMIC_MODELS or model in DINOV2_MODELS:
                if model not in models_to_run:
                    models_to_run.append(model)

    print("="*70)
    print("BENCHMARK CONFIGURATION")
    print("="*70)
    print(f"Image directory: {args.image_dir}")
    print(f"Backend: {args.backend}")
    print(f"Models to test: {', '.join(models_to_run)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Output file: {args.output}")
    print(f"Skip Nomic full mode: {args.skip_nomic_full}")
    print(f"Ntfy topic: {args.ntfy_topic if args.ntfy_topic else 'None (disabled)'}")
    print("="*70)

    # Send start notification
    if args.ntfy_topic:
        send_ntfy_notification(
            args.ntfy_topic,
            f"Starting benchmark suite\nModels: {len(models_to_run)}\nBatch sizes: {len(args.batch_sizes)}",
            title="Benchmark Started",
            priority="default",
            tags=["rocket"]
        )

    # Run benchmarks
    results = []
    total_runs = 0

    # Count total runs
    for model in models_to_run:
        is_nomic = model in NOMIC_MODELS
        for batch_size in args.batch_sizes:
            if is_nomic:
                if not args.skip_nomic_full:
                    total_runs += 1  # Full mode
                total_runs += 1  # Image-only mode
            else:
                total_runs += 1  # DINOv2 (only one mode)

    current_run = 0

    for model in models_to_run:
        is_nomic = model in NOMIC_MODELS

        for batch_size in args.batch_sizes:
            if is_nomic:
                # Run Nomic models in both modes
                if not args.skip_nomic_full:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}] Running {model} batch={batch_size} mode=full")
                    result = run_benchmark(
                        model, batch_size, args.image_dir,
                        backend=args.backend, image_only=False,
                        ntfy_topic=args.ntfy_topic
                    )
                    results.append(result)

                    # Save intermediate results
                    with open(args.output, 'w') as f:
                        f.write(format_results_as_markdown(results))

                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running {model} batch={batch_size} mode=image-only")
                result = run_benchmark(
                    model, batch_size, args.image_dir,
                    backend=args.backend, image_only=True,
                    ntfy_topic=args.ntfy_topic
                )
                results.append(result)

            else:
                # Run DINOv2 models (no image-only mode needed)
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running {model} batch={batch_size}")
                result = run_benchmark(
                    model, batch_size, args.image_dir,
                    backend=args.backend, image_only=False,
                    ntfy_topic=args.ntfy_topic
                )
                results.append(result)

            # Save intermediate results after each run
            with open(args.output, 'w') as f:
                f.write(format_results_as_markdown(results))

    # Write final results
    markdown_output = format_results_as_markdown(results)
    with open(args.output, 'w') as f:
        f.write(markdown_output)

    # Calculate summary stats
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print(f"Results written to: {args.output}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*70)

    # Send completion notification
    if args.ntfy_topic:
        if failed == 0:
            send_ntfy_notification(
                args.ntfy_topic,
                f"All benchmarks completed successfully!\nTotal: {len(results)}\nResults: {args.output}",
                title="Benchmark Suite Complete",
                priority="high",
                tags=["tada", "white_check_mark"]
            )
        else:
            send_ntfy_notification(
                args.ntfy_topic,
                f"Benchmark suite finished with errors\nSuccessful: {successful}\nFailed: {failed}\nResults: {args.output}",
                title="Benchmark Suite Complete (with errors)",
                priority="high",
                tags=["warning", "x"]
            )


if __name__ == "__main__":
    main()
