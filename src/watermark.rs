use crate::core::WaterMarkCore;
use crate::utils::{bits_to_string, shuffle_wm_bits, string_to_bits, unshuffle_wm_bits};
use image::{ImageBuffer, Rgb, Rgba};
use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct WaterMark {
    password_wm: u64,
    core: WaterMarkCore,
    wm_bit: Vec<bool>,
    img: Option<Array3<u8>>,
    /// If set, images are resized to this max dimension before embedding/extracting.
    /// This enables fast mode with consistent extraction regardless of input size.
    embed_size: Option<u32>,
    /// Original image dimensions, stored for upscaling after embed
    original_size: Option<(u32, u32)>,
}

#[pymethods]
impl WaterMark {
    #[new]
    #[pyo3(signature = (password_wm=1, password_img=1, embed_size=None))]
    fn new(password_wm: u64, password_img: u64, embed_size: Option<u32>) -> Self {
        Self {
            password_wm,
            core: WaterMarkCore::new(password_img),
            wm_bit: Vec::new(),
            img: None,
            embed_size,
            original_size: None,
        }
    }

    /// Read image from file path
    fn read_img(&mut self, filename: &str) -> PyResult<()> {
        let img = image::open(filename)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();

        let mut arr = Array3::<u8>::zeros((h as usize, w as usize, 3));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            // Store as BGR to match OpenCV convention
            arr[[y as usize, x as usize, 0]] = pixel[2]; // B
            arr[[y as usize, x as usize, 1]] = pixel[1]; // G
            arr[[y as usize, x as usize, 2]] = pixel[0]; // R
        }

        // Store original size and resize if embed_size is set
        self.original_size = Some((w, h));
        let arr = if let Some(max_dim) = self.embed_size {
            resize_array(&arr, max_dim)
        } else {
            arr
        };

        self.core.read_img_arr(&arr);
        self.img = Some(arr);
        Ok(())
    }

    /// Read image from numpy array (expects BGR format like OpenCV)
    fn read_img_array(&mut self, img: PyReadonlyArray3<u8>) -> PyResult<()> {
        let arr = img.as_array();
        let shape = arr.shape();
        let mut owned = Array3::<u8>::zeros((shape[0], shape[1], shape[2]));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    owned[[i, j, k]] = arr[[i, j, k]];
                }
            }
        }

        // Store original size (w, h) and resize if embed_size is set
        self.original_size = Some((shape[1] as u32, shape[0] as u32));
        let owned = if let Some(max_dim) = self.embed_size {
            resize_array(&owned, max_dim)
        } else {
            owned
        };

        self.core.read_img_arr(&owned);
        self.img = Some(owned);
        Ok(())
    }

    /// Read watermark content
    /// mode: 'img' for image file, 'str' for string, 'bit' for bit array
    #[pyo3(signature = (wm_content, mode="str"))]
    fn read_wm(&mut self, wm_content: &Bound<'_, PyAny>, mode: &str) -> PyResult<()> {
        self.wm_bit = match mode {
            "img" => {
                let filename: String = wm_content.extract()?;
                let img = image::open(&filename)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                let gray = img.to_luma8();

                gray.pixels().map(|p| p.0[0] > 128).collect()
            }
            "str" => {
                let s: String = wm_content.extract()?;
                string_to_bits(&s)
            }
            "bit" => {
                if let Ok(list) = wm_content.downcast::<PyList>() {
                    list.iter()
                        .map(|item| item.extract::<bool>().unwrap_or(false))
                        .collect()
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "bit mode requires a list of booleans",
                    ));
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "mode must be 'img', 'str', or 'bit'",
                ));
            }
        };

        self.wm_bit = shuffle_wm_bits(&self.wm_bit, self.password_wm);
        Ok(())
    }

    /// Get watermark size in bits
    fn wm_size(&self) -> usize {
        self.wm_bit.len()
    }

    /// Embed watermark into image
    /// Returns numpy array if filename is None, otherwise saves to file
    #[pyo3(signature = (filename=None))]
    fn embed<'py>(
        &mut self,
        py: Python<'py>,
        filename: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        if self.img.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No image loaded. Call read_img or read_img_array first.",
            ));
        }

        if self.wm_bit.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No watermark loaded. Call read_wm first.",
            ));
        }

        let result = self.core.embed(&self.wm_bit);

        // Upscale back to original size if embed_size was used
        let result = if self.embed_size.is_some() {
            if let Some((orig_w, orig_h)) = self.original_size {
                resize_array_to(&result, orig_w, orig_h)
            } else {
                result
            }
        } else {
            result
        };

        if let Some(path) = filename {
            save_image(&result, path)?;
        }

        Ok(result.to_pyarray_bound(py).to_owned())
    }

    /// Extract watermark from image
    /// filename: path to watermarked image (or None if using embed_img)
    /// wm_shape: size of watermark (number of bits for str/bit mode, or (h, w) tuple for img mode)
    /// mode: 'img', 'str', or 'bit'
    #[pyo3(signature = (filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode="str"))]
    fn extract(
        &mut self,
        py: Python<'_>,
        filename: Option<&str>,
        embed_img: Option<PyReadonlyArray3<u8>>,
        wm_shape: Option<&Bound<'_, PyAny>>,
        out_wm_name: Option<&str>,
        mode: &str,
    ) -> PyResult<PyObject> {
        let img = if let Some(path) = filename {
            load_image(path)?
        } else if let Some(arr) = embed_img {
            let a = arr.as_array();
            let shape = a.shape();
            let mut owned = Array3::<u8>::zeros((shape[0], shape[1], shape[2]));
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        owned[[i, j, k]] = a[[i, j, k]];
                    }
                }
            }
            owned
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Either filename or embed_img must be provided",
            ));
        };

        // Downscale to embed_size if set
        let img = if let Some(max_dim) = self.embed_size {
            resize_array(&img, max_dim)
        } else {
            img
        };

        let (wm_size, wm_dims) = parse_wm_shape(wm_shape)?;
        let wm_avg = if mode == "str" || mode == "bit" {
            let bits = self.core.extract_with_kmeans(&img, wm_size);
            unshuffle_wm_bits(&bits, self.password_wm, wm_size)
        } else {
            let avg = self.core.extract(&img, wm_size);
            let bits: Vec<bool> = avg.iter().map(|&x| x >= 0.5).collect();
            unshuffle_wm_bits(&bits, self.password_wm, wm_size)
        };

        match mode {
            "str" => {
                let s = bits_to_string(&wm_avg);
                Ok(s.to_object(py))
            }
            "bit" => {
                let list = PyList::new_bound(py, wm_avg.iter().copied());
                Ok(list.to_object(py))
            }
            "img" => {
                if let Some((h, w)) = wm_dims {
                    let mut img_arr = Array3::<u8>::zeros((h, w, 1));
                    for (i, &bit) in wm_avg.iter().enumerate() {
                        if i < h * w {
                            img_arr[[i / w, i % w, 0]] = if bit { 255 } else { 0 };
                        }
                    }

                    if let Some(path) = out_wm_name {
                        let img_data: Vec<u8> = wm_avg
                            .iter()
                            .take(h * w)
                            .map(|&bit| if bit { 255u8 } else { 0u8 })
                            .collect();
                        let img_buf: ImageBuffer<image::Luma<u8>, _> =
                            ImageBuffer::from_raw(w as u32, h as u32, img_data).ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err(
                                    "Failed to create image buffer",
                                )
                            })?;
                        img_buf.save(path).map_err(|e| {
                            pyo3::exceptions::PyIOError::new_err(e.to_string())
                        })?;
                    }

                    Ok(img_arr.to_pyarray_bound(py).to_object(py))
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "img mode requires wm_shape as (height, width) tuple",
                    ))
                }
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be 'img', 'str', or 'bit'",
            )),
        }
    }
}

fn load_image(path: &str) -> PyResult<Array3<u8>> {
    let img = image::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    let mut arr = Array3::<u8>::zeros((h as usize, w as usize, 3));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        // BGR format
        arr[[y as usize, x as usize, 0]] = pixel[2];
        arr[[y as usize, x as usize, 1]] = pixel[1];
        arr[[y as usize, x as usize, 2]] = pixel[0];
    }

    Ok(arr)
}

fn save_image(arr: &Array3<u8>, path: &str) -> PyResult<()> {
    let (h, w, c) = arr.dim();

    if c == 3 {
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                // Convert BGR to RGB
                img.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([arr[[y, x, 2]], arr[[y, x, 1]], arr[[y, x, 0]]]),
                );
            }
        }
        img.save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    } else if c == 4 {
        let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(
                    x as u32,
                    y as u32,
                    Rgba([arr[[y, x, 2]], arr[[y, x, 1]], arr[[y, x, 0]], arr[[y, x, 3]]]),
                );
            }
        }
        img.save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    Ok(())
}

fn parse_wm_shape(wm_shape: Option<&Bound<'_, PyAny>>) -> PyResult<(usize, Option<(usize, usize)>)> {
    match wm_shape {
        None => Err(pyo3::exceptions::PyValueError::new_err("wm_shape is required")),
        Some(shape) => {
            if let Ok(size) = shape.extract::<usize>() {
                return Ok((size, None));
            }
            if let Ok(tuple) = shape.extract::<(usize, usize)>() {
                return Ok((tuple.0 * tuple.1, Some(tuple)));
            }
            if let Ok(list) = shape.extract::<Vec<usize>>() {
                if list.len() == 2 {
                    return Ok((list[0] * list[1], Some((list[0], list[1]))));
                } else if list.len() == 1 {
                    return Ok((list[0], None));
                }
            }

            Err(pyo3::exceptions::PyValueError::new_err(
                "wm_shape must be an integer or (height, width) tuple",
            ))
        }
    }
}

/// Resize array so the longest dimension equals max_dim, preserving aspect ratio
fn resize_array(arr: &Array3<u8>, max_dim: u32) -> Array3<u8> {
    let (h, w, _) = arr.dim();
    let (h, w) = (h as u32, w as u32);

    // Calculate new dimensions preserving aspect ratio
    let (new_w, new_h) = if w >= h {
        let new_w = max_dim;
        let new_h = (h as f64 * max_dim as f64 / w as f64).round() as u32;
        (new_w, new_h.max(1))
    } else {
        let new_h = max_dim;
        let new_w = (w as f64 * max_dim as f64 / h as f64).round() as u32;
        (new_w.max(1), new_h)
    };

    resize_array_to(arr, new_w, new_h)
}

/// Resize array to exact dimensions using SIMD-optimized fast_image_resize
fn resize_array_to(arr: &Array3<u8>, new_w: u32, new_h: u32) -> Array3<u8> {
    use fast_image_resize as fir;

    let (h, w, c) = arr.dim();

    if new_w == w as u32 && new_h == h as u32 {
        return arr.clone();
    }

    // Get contiguous slice of source data (already in BGR order, but fir doesn't care)
    let src_data: Vec<u8> = arr.iter().cloned().collect();

    let src_image = fir::images::Image::from_vec_u8(
        w as u32,
        h as u32,
        src_data,
        fir::PixelType::U8x3,
    )
    .unwrap();

    let mut dst_image = fir::images::Image::new(new_w, new_h, fir::PixelType::U8x3);

    let mut resizer = fir::Resizer::new();
    resizer
        .resize(&src_image, &mut dst_image, None)
        .unwrap();

    // Convert back to Array3
    let dst_data = dst_image.into_vec();
    Array3::from_shape_vec((new_h as usize, new_w as usize, c), dst_data).unwrap()
}
