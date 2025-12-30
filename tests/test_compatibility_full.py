#!/usr/bin/env python3
"""
Thorough compatibility tests (~4 minutes with -n auto).

Run with: pytest tests/test_compatibility_full.py -n auto
"""
import numpy as np
import pytest
from blind_watermark import WaterMark as PyWaterMark
from blind_watermark import cut_att3 as py_cut_att3, resize_att as py_resize_att, bright_att as py_bright_att, rot_att as py_rot_att
from blazing_blind_watermark import WaterMark as RustWaterMark
from blazing_blind_watermark import cut_att3, resize_att, bright_att, rot_att

REPEATS = 5
IMAGE_SIZES = [
    (256, 256),
    (512, 512),
    (640, 480),
    (1920, 1080),
    (3840, 2160),
]
PASSWORDS = [(1, 1), (12345, 67890)]


class TestCrossExtractionStr:
    """Test that Rust can extract Python's watermarks and vice versa (string mode)."""

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    @pytest.mark.parametrize("passwords", PASSWORDS)
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_rust_extracts_python_str(self, size, passwords, repeat):
        np.random.seed((repeat * 1000 + size[0] + passwords[0]) % (2**31))
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        wm_text = f"test{repeat}"
        pw_wm, pw_img = passwords

        py_bwm = PyWaterMark(password_wm=pw_wm, password_img=pw_img)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_text, mode='str')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed().astype(np.uint8)

        rust_bwm = RustWaterMark(password_wm=pw_wm, password_img=pw_img)
        rust_extracted = rust_bwm.extract(embed_img=py_embedded, wm_shape=wm_len, mode='str')

        assert rust_extracted == wm_text

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    @pytest.mark.parametrize("passwords", PASSWORDS)
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_python_extracts_rust_str(self, size, passwords, repeat):
        np.random.seed((repeat * 1000 + size[0] * 10 + passwords[0]) % (2**31))
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        wm_text = f"test{repeat}"
        pw_wm, pw_img = passwords

        rust_bwm = RustWaterMark(password_wm=pw_wm, password_img=pw_img)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_text, mode='str')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        py_bwm = PyWaterMark(password_wm=pw_wm, password_img=pw_img)
        py_extracted = py_bwm.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='str')

        assert py_extracted == wm_text


class TestCrossExtractionBit:
    """Test that Rust can extract Python's watermarks and vice versa (bit mode)."""

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    @pytest.mark.parametrize("passwords", PASSWORDS)
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_rust_extracts_python_bit(self, size, passwords, repeat):
        np.random.seed((repeat * 1000 + size[0] * 10 + passwords[0] + 1) % (2**31))
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        wm_bits = [bool(b) for b in np.random.randint(0, 2, 32)]
        pw_wm, pw_img = passwords

        py_bwm = PyWaterMark(password_wm=pw_wm, password_img=pw_img)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_bits, mode='bit')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed().astype(np.uint8)

        rust_bwm = RustWaterMark(password_wm=pw_wm, password_img=pw_img)
        rust_extracted = rust_bwm.extract(embed_img=py_embedded, wm_shape=wm_len, mode='bit')

        assert list(rust_extracted) == wm_bits

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    @pytest.mark.parametrize("passwords", PASSWORDS)
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_python_extracts_rust_bit(self, size, passwords, repeat):
        np.random.seed((repeat * 1000 + size[0] * 10 + passwords[0] + 2) % (2**31))
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        wm_bits = [bool(b) for b in np.random.randint(0, 2, 32)]
        pw_wm, pw_img = passwords

        rust_bwm = RustWaterMark(password_wm=pw_wm, password_img=pw_img)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_bits, mode='bit')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        py_bwm = PyWaterMark(password_wm=pw_wm, password_img=pw_img)
        py_extracted = py_bwm.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='bit')

        assert list(py_extracted) == wm_bits


class TestSelfExtraction:
    """Test that each implementation can extract its own watermarks."""

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_rust_self_extract_str(self, size, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        wm_text = f"self{repeat}"

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_text, mode='str')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        rust_bwm2 = RustWaterMark(password_wm=1, password_img=1)
        rust_extracted = rust_bwm2.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='str')

        assert rust_extracted == wm_text

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_python_self_extract_str(self, size, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        wm_text = f"self{repeat}"

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_text, mode='str')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed()

        py_bwm2 = PyWaterMark(password_wm=1, password_img=1)
        py_extracted = py_bwm2.extract(embed_img=py_embedded, wm_shape=wm_len, mode='str')

        assert py_extracted == wm_text


class TestAttacksCutAtt3:
    """Test cut_att3 attack produces identical results."""

    @pytest.mark.parametrize("size", IMAGE_SIZES[:3])  # Skip large sizes for speed
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_cut_att3_no_scale(self, size, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        h, w = size
        loc = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

        py_result = py_cut_att3(input_img=img, loc=loc)
        rust_result = cut_att3(input_img=img, loc=loc)

        assert py_result.shape == rust_result.shape
        np.testing.assert_array_equal(py_result.astype(np.uint8), rust_result)


class TestAttacksResizeAtt:
    """Test resize_att attack produces identical results."""

    @pytest.mark.parametrize("size", IMAGE_SIZES[:3])
    @pytest.mark.parametrize("out_shape", [(64, 64), (128, 128), (256, 256)])
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_resize_att(self, size, out_shape, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        py_result = py_resize_att(input_img=img, out_shape=out_shape)
        rust_result = resize_att(input_img=img, out_shape=out_shape)

        assert py_result.shape == rust_result.shape
        np.testing.assert_array_equal(py_result, rust_result)


class TestAttacksBrightAtt:
    """Test bright_att attack produces identical results."""

    @pytest.mark.parametrize("size", IMAGE_SIZES[:3])
    @pytest.mark.parametrize("ratio", [0.5, 0.8, 1.0, 1.2])
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_bright_att(self, size, ratio, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        py_result = py_bright_att(input_img=img, ratio=ratio)
        rust_result = bright_att(input_img=img, ratio=ratio)

        assert py_result.shape == rust_result.shape
        np.testing.assert_array_equal(py_result, rust_result)


class TestAttacksRotAtt:
    """Test rot_att attack produces identical results."""

    @pytest.mark.parametrize("size", IMAGE_SIZES[:3])
    @pytest.mark.parametrize("angle", [0, 45, 90, 180])
    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_rot_att(self, size, angle, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)

        py_result = py_rot_att(input_img=img, angle=angle)
        rust_result = rot_att(input_img=img, angle=angle)

        assert py_result.shape == rust_result.shape
        np.testing.assert_array_equal(py_result, rust_result)


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_single_char_cross(self, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_text = "x"

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_text, mode='str')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_extracted = py_bwm.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='str')

        assert py_extracted == wm_text

    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_odd_dimensions_cross(self, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (127, 255, 3), dtype=np.uint8)
        wm_text = "odd"

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_text, mode='str')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed().astype(np.uint8)

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_extracted = rust_bwm.extract(embed_img=py_embedded, wm_shape=wm_len, mode='str')

        assert rust_extracted == wm_text

    @pytest.mark.parametrize("repeat", range(REPEATS))
    def test_medium_watermark(self, repeat):
        np.random.seed(repeat)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        wm_text = "hello world"

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_text, mode='str')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_extracted = py_bwm.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='str')

        assert py_extracted == wm_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
