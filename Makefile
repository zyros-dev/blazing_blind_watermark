.PHONY: setup build dev test test-full bench bench-quick clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install maturin pytest numpy opencv-python blind_watermark pandas seaborn matplotlib py-spy

build:
	$(VENV)/bin/maturin build --release

dev:
	$(VENV)/bin/maturin develop --release

test: dev
	$(PYTHON) -m pytest tests/test_compatibility.py -v

test-full: dev
	$(PYTHON) -m pytest tests/test_compatibility_full.py -v -n auto

bench: dev
	$(PYTHON) benchmarks/benchmark.py

bench-quick: dev
	$(PYTHON) benchmarks/quick_bench.py

cargo-build:
	cargo build --release

cargo-test:
	cargo test

cargo-check:
	cargo check

cargo-clippy:
	cargo clippy

clean:
	rm -rf target/
	rm -rf $(VENV)
	rm -rf *.egg-info
	rm -rf benchmarks/*.svg benchmarks/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
