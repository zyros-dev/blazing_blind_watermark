#!/usr/bin/env python3
"""Quick benchmark with flame chart for large image profiling."""
import subprocess
import shutil
import tempfile
from pathlib import Path

SIZE = (8192, 8192)

def main():
    py_spy = shutil.which("py-spy")
    if not py_spy:
        print("py-spy not found (pip install py-spy)")
        return

    print(f"Profiling {SIZE[0]}x{SIZE[1]} (single embed + extract)...")

    script = f"""
import time
import numpy as np
from blazing_blind_watermark import WaterMark

img = np.random.randint(0, 255, {SIZE + (3,)}, dtype=np.uint8)
wm = "test watermark"

start = time.perf_counter()

bwm = WaterMark(password_img=1, password_wm=1)
bwm.read_img_array(img)
bwm.read_wm(wm, mode='str')
embedded = bwm.embed()

bwm2 = WaterMark(password_img=1, password_wm=1)
bwm2.extract(embed_img=embedded, wm_shape=bwm.wm_size(), mode='str')

print(f"Total: {{time.perf_counter() - start:.1f}}s")
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    out_svg = Path(__file__).parent / "flamegraph_8k.svg"

    try:
        result = subprocess.run(
            [py_spy, "record", "-o", str(out_svg), "--native", "--", "python", script_path],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            print(f"Saved: {out_svg}")
        else:
            print(f"py-spy failed: {result.stderr}")
    finally:
        Path(script_path).unlink()

if __name__ == "__main__":
    main()
