#!/usr/bin/env python3
"""
Benchmark comparing Python blind_watermark vs Rust blazing_blind_watermark.
Generates box plots and saves raw data to CSV.
"""
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ITERATIONS = 10
WARMUP = 2

# All sizes to benchmark (Rust only beyond PYTHON_CUTOFF)
SIZES = [
    (128, 128),
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1536, 1536),
    (2048, 2048),
    (3072, 3072),
    (4096, 4096),
    (6144, 6144),
    (8192, 8192),
]

PYTHON_CUTOFF = 2048  # Don't run Python beyond this size


def collect_timings(func, iterations=ITERATIONS, warmup=WARMUP):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return times


def main():
    try:
        from blind_watermark import WaterMark as PyWaterMark
        has_python = True
    except ImportError:
        print("Python blind_watermark not installed")
        has_python = False

    try:
        from blazing_blind_watermark import WaterMark as RustWaterMark
        has_rust = True
    except ImportError:
        print("Rust blazing_blind_watermark not installed")
        has_rust = False

    if not has_rust:
        return

    wm_text = "Hello, this is a test watermark!"
    records = []
    out_dir = Path(__file__).parent

    for size in SIZES:
        label = f"{size[0]}x{size[1]}"
        run_python = has_python and size[0] <= PYTHON_CUTOFF
        print(f"Benchmarking {label}..." + ("" if run_python else " (Rust only)"))

        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        # Python
        if run_python:
            def py_embed():
                bwm = PyWaterMark(password_img=1, password_wm=1)
                bwm.read_img(img=img)
                bwm.read_wm(wm_text, mode='str')
                return bwm.embed()

            for t in collect_timings(py_embed):
                records.append({"size": label, "pixels": size[0] * size[1], "impl": "Python", "op": "embed", "time_ms": t})

            py_result = py_embed()
            bwm_tmp = PyWaterMark(password_img=1, password_wm=1)
            bwm_tmp.read_img(img=img)
            bwm_tmp.read_wm(wm_text, mode='str')
            py_wm_len = len(bwm_tmp.wm_bit)

            def py_extract():
                bwm = PyWaterMark(password_img=1, password_wm=1)
                return bwm.extract(embed_img=py_result, wm_shape=py_wm_len, mode='str')

            for t in collect_timings(py_extract):
                records.append({"size": label, "pixels": size[0] * size[1], "impl": "Python", "op": "extract", "time_ms": t})

        # Rust
        def rust_embed():
            bwm = RustWaterMark(password_img=1, password_wm=1)
            bwm.read_img_array(img)
            bwm.read_wm(wm_text, mode='str')
            return bwm.embed()

        for t in collect_timings(rust_embed):
            records.append({"size": label, "pixels": size[0] * size[1], "impl": "Rust", "op": "embed", "time_ms": t})

        rust_result = rust_embed()
        bwm_tmp = RustWaterMark(password_img=1, password_wm=1)
        bwm_tmp.read_img_array(img)
        bwm_tmp.read_wm(wm_text, mode='str')
        rust_wm_len = bwm_tmp.wm_size()

        def rust_extract():
            bwm = RustWaterMark(password_img=1, password_wm=1)
            return bwm.extract(embed_img=rust_result, wm_shape=rust_wm_len, mode='str')

        for t in collect_timings(rust_extract):
            records.append({"size": label, "pixels": size[0] * size[1], "impl": "Rust", "op": "extract", "time_ms": t})

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = out_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved data to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary (median times in ms)")
    print("=" * 60)
    summary = df.groupby(["size", "impl", "op"])["time_ms"].median().unstack(["impl", "op"])
    print(summary.to_string())

    # Speedup for sizes with both implementations
    print("\n" + "=" * 60)
    print("Speedup (Python / Rust)")
    print("=" * 60)
    for size in SIZES:
        if size[0] > PYTHON_CUTOFF:
            break
        label = f"{size[0]}x{size[1]}"
        for op in ["embed", "extract"]:
            py_data = df[(df["size"] == label) & (df["impl"] == "Python") & (df["op"] == op)]["time_ms"]
            rust_data = df[(df["size"] == label) & (df["impl"] == "Rust") & (df["op"] == op)]["time_ms"]
            if not py_data.empty and not rust_data.empty:
                print(f"{label} {op}: {py_data.median() / rust_data.median():.1f}x")

    # Create comparison plot (only sizes with Python data)
    compare_sizes = [f"{s[0]}x{s[1]}" for s in SIZES if s[0] <= PYTHON_CUTOFF]
    df_compare = df[df["size"].isin(compare_sizes)]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, op in zip(axes, ["embed", "extract"]):
        subset = df_compare[df_compare["op"] == op]
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

    plot_path = out_dir / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {plot_path}")

    # Create Rust scaling plot (all sizes)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df_rust = df[df["impl"] == "Rust"]

    sns.boxplot(
        data=df_rust,
        x="size",
        y="time_ms",
        hue="op",
        ax=ax2,
        palette={"embed": "#DEA584", "extract": "#B5651D"},
    )
    ax2.set_title("Rust Performance Scaling", fontsize=14)
    ax2.set_xlabel("Image Size", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title="Operation")
    plt.tight_layout()

    scaling_path = out_dir / "benchmark_scaling.png"
    plt.savefig(scaling_path, dpi=150, bbox_inches="tight")
    print(f"Saved scaling plot to {scaling_path}")

    plt.show()


if __name__ == "__main__":
    main()
