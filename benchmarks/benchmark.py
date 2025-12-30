#!/usr/bin/env python3
"""
Benchmark comparing Python blind_watermark vs Rust blazing_blind_watermark.
Generates box plots showing latency distributions.
"""
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ITERATIONS = 50
WARMUP = 3
SIZES = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4196, 4196), (8192, 8192)]


def collect_timings(func, iterations=ITERATIONS, warmup=WARMUP):
    """Run function multiple times and return list of times in milliseconds."""
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    return times


def main():
    try:
        from blind_watermark import WaterMark as PyWaterMark
        has_python = True
    except ImportError:
        print("Python blind_watermark not installed (pip install blind-watermark)")
        has_python = False

    try:
        from blazing_blind_watermark import WaterMark as RustWaterMark
        has_rust = True
    except ImportError:
        print("Rust blazing_blind_watermark not installed (maturin develop --release)")
        has_rust = False

    if not has_rust:
        return

    wm_text = "Hello, this is a test watermark!"
    records = []

    for size in SIZES:
        label = f"{size[0]}x{size[1]}"
        print(f"Benchmarking {label}...")

        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        # Python embed
        if has_python:
            def py_embed():
                bwm = PyWaterMark(password_img=1, password_wm=1)
                bwm.read_img(img=img)
                bwm.read_wm(wm_text, mode='str')
                return bwm.embed()

            for t in collect_timings(py_embed):
                records.append({"size": label, "impl": "Python", "op": "embed", "time_ms": t})

            # Get embedded image and wm_len for extraction
            py_result = py_embed()
            bwm_tmp = PyWaterMark(password_img=1, password_wm=1)
            bwm_tmp.read_img(img=img)
            bwm_tmp.read_wm(wm_text, mode='str')
            py_wm_len = len(bwm_tmp.wm_bit)

            def py_extract():
                bwm = PyWaterMark(password_img=1, password_wm=1)
                return bwm.extract(embed_img=py_result, wm_shape=py_wm_len, mode='str')

            for t in collect_timings(py_extract):
                records.append({"size": label, "impl": "Python", "op": "extract", "time_ms": t})

        # Rust embed
        if has_rust:
            def rust_embed():
                bwm = RustWaterMark(password_img=1, password_wm=1)
                bwm.read_img_array(img)
                bwm.read_wm(wm_text, mode='str')
                return bwm.embed()

            for t in collect_timings(rust_embed):
                records.append({"size": label, "impl": "Rust", "op": "embed", "time_ms": t})

            rust_result = rust_embed()
            bwm_tmp = RustWaterMark(password_img=1, password_wm=1)
            bwm_tmp.read_img_array(img)
            bwm_tmp.read_wm(wm_text, mode='str')
            rust_wm_len = bwm_tmp.wm_size()

            def rust_extract():
                bwm = RustWaterMark(password_img=1, password_wm=1)
                return bwm.extract(embed_img=rust_result, wm_shape=rust_wm_len, mode='str')

            for t in collect_timings(rust_extract):
                records.append({"size": label, "impl": "Rust", "op": "extract", "time_ms": t})

    df = pd.DataFrame(records)

    # Print summary stats
    print("\n" + "=" * 60)
    print("Summary (median times)")
    print("=" * 60)
    summary = df.groupby(["size", "impl", "op"])["time_ms"].median().unstack(["impl", "op"])
    print(summary.to_string())

    if has_python and has_rust:
        print("\n" + "=" * 60)
        print("Speedup (Python / Rust)")
        print("=" * 60)
        for size in SIZES:
            label = f"{size[0]}x{size[1]}"
            for op in ["embed", "extract"]:
                py_med = df[(df["size"] == label) & (df["impl"] == "Python") & (df["op"] == op)]["time_ms"].median()
                rust_med = df[(df["size"] == label) & (df["impl"] == "Rust") & (df["op"] == op)]["time_ms"].median()
                print(f"{label} {op}: {py_med / rust_med:.1f}x")

    # Create plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, op in zip(axes, ["embed", "extract"]):
        subset = df[df["op"] == op]
        sns.boxplot(
            data=subset,
            x="size",
            y="time_ms",
            hue="impl",
            ax=ax,
            palette={"Python": "#3572A5", "Rust": "#DEA584"},
        )
        ax.set_title(f"{op.capitalize()} Latency", fontsize=14)
        ax.set_xlabel("Image Size", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.legend(title="Implementation")

    plt.suptitle("blazing_blind_watermark vs blind_watermark", fontsize=16, y=1.02)
    plt.tight_layout()

    out_path = Path(__file__).parent / "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
