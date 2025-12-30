#!/usr/bin/env python3
"""
Quick compatibility sanity check (~10 seconds).
For thorough testing, run test_compatibility_full.py
"""
import numpy as np
import pytest
from blind_watermark import WaterMark as PyWaterMark
from blazing_blind_watermark import WaterMark as RustWaterMark


class TestCrossCompatibility:
    """Core cross-extraction tests."""

    def test_rust_extracts_python_str(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_text = "hello"

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_text, mode='str')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed().astype(np.uint8)

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_extracted = rust_bwm.extract(embed_img=py_embedded, wm_shape=wm_len, mode='str')

        assert rust_extracted == wm_text

    def test_python_extracts_rust_str(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_text = "world"

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_text, mode='str')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_extracted = py_bwm.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='str')

        assert py_extracted == wm_text

    def test_rust_extracts_python_bit(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_bits = [True, False, True, True, False, False, True, False]

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_bits, mode='bit')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed().astype(np.uint8)

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_extracted = rust_bwm.extract(embed_img=py_embedded, wm_shape=wm_len, mode='bit')

        assert list(rust_extracted) == wm_bits

    def test_python_extracts_rust_bit(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_bits = [False, True, False, True, True, True, False, True]

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_bits, mode='bit')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_extracted = py_bwm.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='bit')

        assert list(py_extracted) == wm_bits


class TestSelfExtraction:
    """Sanity check: each implementation extracts its own watermarks."""

    def test_rust_self_extract(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_text = "test"

        rust_bwm = RustWaterMark(password_wm=1, password_img=1)
        rust_bwm.read_img_array(img)
        rust_bwm.read_wm(wm_text, mode='str')
        wm_len = rust_bwm.wm_size()
        rust_embedded = rust_bwm.embed()

        rust_bwm2 = RustWaterMark(password_wm=1, password_img=1)
        rust_extracted = rust_bwm2.extract(embed_img=rust_embedded, wm_shape=wm_len, mode='str')

        assert rust_extracted == wm_text

    def test_python_self_extract(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        wm_text = "test"

        py_bwm = PyWaterMark(password_wm=1, password_img=1)
        py_bwm.read_img(img=img)
        py_bwm.read_wm(wm_text, mode='str')
        wm_len = len(py_bwm.wm_bit)
        py_embedded = py_bwm.embed()

        py_bwm2 = PyWaterMark(password_wm=1, password_img=1)
        py_extracted = py_bwm2.extract(embed_img=py_embedded, wm_shape=wm_len, mode='str')

        assert py_extracted == wm_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
