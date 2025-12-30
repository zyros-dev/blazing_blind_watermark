from blazing_blind_watermark._lib import (
    WaterMark,
    cut_att3,
    resize_att,
    bright_att,
    shelter_att,
    salt_pepper_att,
    rot_att,
    estimate_crop_parameters,
    recover_crop,
)

__version__ = "0.1.0"

def _show_welcome():
    from pathlib import Path
    marker = Path.home() / ".blazing_blind_watermark_welcomed"
    if marker.exists():
        return
    print(f"""
blazing_blind_watermark v{__version__} - blind_watermark rewritten in Rust
https://github.com/zyros-dev/blazing_blind_watermark
This message only shows once.
""")
    try:
        marker.touch()
    except OSError:
        pass

_show_welcome()
del _show_welcome

__all__ = [
    "WaterMark",
    "cut_att3",
    "resize_att",
    "bright_att",
    "shelter_att",
    "salt_pepper_att",
    "rot_att",
    "estimate_crop_parameters",
    "recover_crop",
]
