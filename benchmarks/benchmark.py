#!/usr/bin/env python3
"""
Benchmark comparing Python blind_watermark vs Rust blazing_blind_watermark.
Generates box plots showing latency distributions and flame graphs.
"""
import time
import subprocess
import shutil
import tempfile
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ITERATIONS = 50
WARMUP = 3
SIZES = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4196, 4196), (8192, 8192)]
PROFILE_ITERATIONS = 20
PROFILE_SIZE = (1024, 1024)


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


def run_profiling(out_dir: Path):
    """Generate flame graphs using py-spy for both implementations."""
    py_spy = shutil.which("py-spy")
    if not py_spy:
        print("\npy-spy not found, skipping flame graphs (pip install py-spy)")
        return

    print("\n" + "=" * 60)
    print("Generating flame graphs...")
    print("=" * 60)

    h, w = PROFILE_SIZE
    iterations = PROFILE_ITERATIONS

    # Script for Python blind_watermark profiling
    py_script = textwrap.dedent(f"""
        import numpy as np
        from blind_watermark import WaterMark

        img = np.random.randint(0, 255, ({h}, {w}, 3), dtype=np.uint8)
        wm_text = "profiling test watermark"

        for _ in range({iterations}):
            bwm = WaterMark(password_img=1, password_wm=1)
            bwm.read_img(img=img)
            bwm.read_wm(wm_text, mode='str')
            embedded = bwm.embed()
            wm_len = len(bwm.wm_bit)

            bwm2 = WaterMark(password_img=1, password_wm=1)
            bwm2.extract(embed_img=embedded, wm_shape=wm_len, mode='str')
    """)

    # Script for Rust blazing_blind_watermark profiling
    rust_script = textwrap.dedent(f"""
        import numpy as np
        from blazing_blind_watermark import WaterMark

        img = np.random.randint(0, 255, ({h}, {w}, 3), dtype=np.uint8)
        wm_text = "profiling test watermark"

        for _ in range({iterations}):
            bwm = WaterMark(password_img=1, password_wm=1)
            bwm.read_img_array(img)
            bwm.read_wm(wm_text, mode='str')
            embedded = bwm.embed()
            wm_len = bwm.wm_size()

            bwm2 = WaterMark(password_img=1, password_wm=1)
            bwm2.extract(embed_img=embedded, wm_shape=wm_len, mode='str')
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Profile Python implementation
        py_script_path = tmpdir / "profile_python.py"
        py_script_path.write_text(py_script)
        py_svg = out_dir / "flamegraph_python.svg"

        print(f"Profiling Python blind_watermark ({iterations} iterations @ {w}x{h})...")
        try:
            subprocess.run(
                [py_spy, "record", "-o", str(py_svg), "--native", "--", "python", str(py_script_path)],
                check=True,
                capture_output=True,
            )
            print(f"  Saved: {py_svg}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e.stderr.decode() if e.stderr else e}")

        # Profile Rust implementation
        rust_script_path = tmpdir / "profile_rust.py"
        rust_script_path.write_text(rust_script)
        rust_svg = out_dir / "flamegraph_rust.svg"

        print(f"Profiling Rust blazing_blind_watermark ({iterations} iterations @ {w}x{h})...")
        try:
            subprocess.run(
                [py_spy, "record", "-o", str(rust_svg), "--native", "--", "python", str(rust_script_path)],
                check=True,
                capture_output=True,
            )
            print(f"  Saved: {rust_svg}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e.stderr.decode() if e.stderr else e}")


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

    out_dir = Path(__file__).parent
    out_path = out_dir / "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()

    # Generate flame graphs
    if has_python and has_rust:
        run_profiling(out_dir)


if __name__ == "__main__":
    main()
