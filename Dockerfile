# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY main.py ./

# Install PyTorch with CUDA support and other dependencies
RUN uv pip install --system torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --system pillow>=10.0.0 transformers>=4.40.0 && \
    uv pip install --system "colpali-engine @ git+https://github.com/illuin-tech/colpali.git"

# Create directory for images
RUN mkdir -p /app/images

# Set entrypoint
ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]
