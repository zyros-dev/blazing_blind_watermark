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

class _BWNotes:
    _marker_path = None

    @classmethod
    def _get_marker(cls):
        if cls._marker_path is None:
            from pathlib import Path
            cls._marker_path = Path.home() / ".blazing_blind_watermark_welcomed"
        return cls._marker_path

    @classmethod
    def close(cls):
        """Disable the welcome message permanently."""
        try:
            cls._get_marker().touch()
        except OSError:
            pass

    @classmethod
    def _show(cls):
        if cls._get_marker().exists():
            return
        print(f"""
blazing_blind_watermark v{__version__} - blind_watermark rewritten in Rust
Your star means a lot! https://github.com/zyros-dev/blazing_blind_watermark
This message only shows once. To close it: `blazing_blind_watermark.bw_notes.close()`
""")
        cls.close()

bw_notes = _BWNotes()
bw_notes._show()

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
