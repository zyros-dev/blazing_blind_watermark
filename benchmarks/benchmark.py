#!/usr/bin/env python3
"""
Benchmark comparing Python blind_watermark vs Rust blazing_blind_watermark
"""
import time
import numpy as np

def benchmark(name, func, iterations=10, warmup=2):
    """Run benchmark and return average time in milliseconds"""
    # Warmup
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"{name}: {avg*1000:.2f}ms avg (+/- {std*1000:.2f}ms)")
    return avg


def main():
    # Try to import both libraries
    try:
        from blind_watermark import WaterMark as PyWaterMark
        has_python = True
    except ImportError:
        print("Python blind_watermark not installed, skipping Python benchmarks")
        print("Install with: pip install blind-watermark")
        has_python = False

    try:
        from blazing_blind_watermark import WaterMark as RustWaterMark
        has_rust = True
    except ImportError:
        print("Rust blazing_blind_watermark not installed")
        print("Build with: maturin develop --release")
        has_rust = False

    if not has_rust:
        return

    wm_text = "Hello, this is a test watermark!"

    # Test with various image sizes
    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Image size: {size[0]}x{size[1]}")
        print('='*50)

        # Create random test image
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        # Python embedding
        if has_python:
            def py_embed():
                bwm = PyWaterMark(password_img=1, password_wm=1)
                bwm.read_img(img=img)
                bwm.read_wm(wm_text, mode='str')
                return bwm.embed()

            py_time = benchmark("Python embed", py_embed)
            py_result = py_embed()
            wm_len = len(PyWaterMark(password_img=1, password_wm=1).read_wm(wm_text, mode='str') or [])

            # Get watermark length for extraction
            bwm_tmp = PyWaterMark(password_img=1, password_wm=1)
            bwm_tmp.read_img(img=img)
            bwm_tmp.read_wm(wm_text, mode='str')
            wm_len = len(bwm_tmp.wm_bit)

            def py_extract():
                bwm = PyWaterMark(password_img=1, password_wm=1)
                return bwm.extract(embed_img=py_result, wm_shape=wm_len, mode='str')

            benchmark("Python extract", py_extract)

        # Rust embedding
        if has_rust:
            def rust_embed():
                bwm = RustWaterMark(password_img=1, password_wm=1)
                bwm.read_img_array(img)
                bwm.read_wm(wm_text, mode='str')
                return bwm.embed()

            rust_time = benchmark("Rust embed", rust_embed)
            rust_result = rust_embed()

            # Get watermark size
            bwm_tmp = RustWaterMark(password_img=1, password_wm=1)
            bwm_tmp.read_img_array(img)
            bwm_tmp.read_wm(wm_text, mode='str')
            rust_wm_len = bwm_tmp.wm_size()

            def rust_extract():
                bwm = RustWaterMark(password_img=1, password_wm=1)
                return bwm.extract(embed_img=rust_result, wm_shape=rust_wm_len, mode='str')

            benchmark("Rust extract", rust_extract)

            # Show speedup
            if has_python:
                speedup = py_time / rust_time
                print(f"\nSpeedup: {speedup:.1f}x faster")

    # Verify correctness
    print(f"\n{'='*50}")
    print("Verification")
    print('='*50)

    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_message = "Test123"

    if has_rust:
        bwm = RustWaterMark(password_img=1, password_wm=1)
        bwm.read_img_array(img)
        bwm.read_wm(test_message, mode='str')
        wm_len = bwm.wm_size()
        embedded = bwm.embed()

        bwm2 = RustWaterMark(password_img=1, password_wm=1)
        extracted = bwm2.extract(embed_img=embedded, wm_shape=wm_len, mode='str')

        print(f"Original: '{test_message}'")
        print(f"Extracted: '{extracted}'")
        print(f"Match: {test_message == extracted}")


if __name__ == "__main__":
    main()
