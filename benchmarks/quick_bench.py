#!/usr/bin/env python3
"""Quick benchmark with flame chart for Rust version."""
import time
import subprocess
import shutil
import tempfile
import numpy as np
from pathlib import Path

SIZE = (1024, 1024)
PROFILE_ITERATIONS = 30

def bench(name, func, iterations=3):
    func()  # warmup
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    avg = sum(times) / len(times)
    print(f"{name}: {avg:.1f}ms")
    return avg

def main():
    from blazing_blind_watermark import WaterMark as RustWaterMark

    img = np.random.randint(0, 255, (*SIZE, 3), dtype=np.uint8)
    wm = "test watermark"

    def rust_embed():
        bwm = RustWaterMark(password_img=1, password_wm=1)
        bwm.read_img_array(img)
        bwm.read_wm(wm, mode='str')
        return bwm.embed()

    print(f"Size: {SIZE[0]}x{SIZE[1]}")
    bench("Rust embed", rust_embed)

    # Generate flame graph
    py_spy = shutil.which("py-spy")
    if not py_spy:
        print("\npy-spy not found (pip install py-spy)")
        return

    print(f"\nGenerating flame graph ({PROFILE_ITERATIONS} iterations)...")
    
    script = f"""
import numpy as np
from blazing_blind_watermark import WaterMark

img = np.random.randint(0, 255, {SIZE + (3,)}, dtype=np.uint8)
wm = "test watermark"

for _ in range({PROFILE_ITERATIONS}):
    bwm = WaterMark(password_img=1, password_wm=1)
    bwm.read_img_array(img)
    bwm.read_wm(wm, mode='str')
    embedded = bwm.embed()
    
    bwm2 = WaterMark(password_img=1, password_wm=1)
    bwm2.extract(embed_img=embedded, wm_shape=bwm.wm_size(), mode='str')
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    out_svg = Path(__file__).parent / "flamegraph_rust.svg"
    
    try:
        result = subprocess.run(
            [py_spy, "record", "-o", str(out_svg), "--native", "--", "python", script_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Saved: {out_svg}")
        else:
            print(f"py-spy failed: {result.stderr}")
    finally:
        Path(script_path).unlink()

if __name__ == "__main__":
    main()
