FROM rust:1.83-bookworm

# Install Python and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv and install Python deps
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip maturin pytest numpy opencv-python blind_watermark

# Copy source
COPY . .

# Build the Rust extension
RUN maturin develop --release

# Default command runs tests
CMD ["pytest", "tests/test_compatibility.py", "-v"]
