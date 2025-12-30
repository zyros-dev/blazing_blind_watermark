use image::{ImageBuffer, Rgb};
use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;

/// Crop and scale attack
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, loc=None, scale=None))]
pub fn cut_att3<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    loc: Option<(usize, usize, usize, usize)>,
    scale: Option<f32>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_input(input_filename, input_img)?;
    let (h, w, c) = img.dim();
    let (x1, y1, x2, y2) = loc.unwrap_or((0, 0, w, h));

    let crop_h = y2 - y1;
    let crop_w = x2 - x1;
    let mut cropped = Array3::<u8>::zeros((crop_h, crop_w, c));
    for i in 0..crop_h {
        for j in 0..crop_w {
            for k in 0..c {
                cropped[[i, j, k]] = img[[y1 + i, x1 + j, k]];
            }
        }
    }

    let output = if let Some(s) = scale {
        if (s - 1.0).abs() > 0.001 {
            resize_array(&cropped, (crop_h as f32 * s) as usize, (crop_w as f32 * s) as usize)
        } else {
            cropped
        }
    } else {
        cropped
    };

    if let Some(path) = output_file_name {
        save_array(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

/// Resize attack
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, out_shape=(500, 500)))]
pub fn resize_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    out_shape: (usize, usize),
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_input(input_filename, input_img)?;

    let output = resize_array(&img, out_shape.1, out_shape.0); // Note: OpenCV uses (w, h)

    if let Some(path) = output_file_name {
        save_array(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

/// Brightness attack
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, ratio=0.8))]
pub fn bright_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    ratio: f32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_input(input_filename, input_img)?;
    let (h, w, c) = img.dim();

    let mut output = Array3::<u8>::zeros((h, w, c));
    for i in 0..h {
        for j in 0..w {
            for k in 0..c {
                let val = (img[[i, j, k]] as f32 * ratio).min(255.0);
                output[[i, j, k]] = val as u8;
            }
        }
    }

    if let Some(path) = output_file_name {
        save_array(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

/// Shelter (occlusion) attack
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, ratio=0.1, n=3))]
pub fn shelter_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    ratio: f32,
    n: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_input(input_filename, input_img)?;
    let (h, w, c) = img.dim();

    let mut output = img.clone();
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let tmp_h = rng.gen::<f32>() * (1.0 - ratio);
        let start_h = (tmp_h * h as f32) as usize;
        let end_h = ((tmp_h + ratio) * h as f32) as usize;

        let tmp_w = rng.gen::<f32>() * (1.0 - ratio);
        let start_w = (tmp_w * w as f32) as usize;
        let end_w = ((tmp_w + ratio) * w as f32) as usize;

        for i in start_h..end_h.min(h) {
            for j in start_w..end_w.min(w) {
                for k in 0..c {
                    output[[i, j, k]] = 255;
                }
            }
        }
    }

    if let Some(path) = output_file_name {
        save_array(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

/// Salt and pepper noise attack
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, ratio=0.01))]
pub fn salt_pepper_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    ratio: f32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_input(input_filename, input_img)?;
    let (h, w, c) = img.dim();

    let mut output = img.clone();
    let mut rng = rand::thread_rng();

    for i in 0..h {
        for j in 0..w {
            if rng.gen::<f32>() < ratio {
                for k in 0..c {
                    output[[i, j, k]] = 255;
                }
            }
        }
    }

    if let Some(path) = output_file_name {
        save_array(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

/// Rotation attack
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, angle=45.0))]
pub fn rot_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    angle: f32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_input(input_filename, input_img)?;
    let (h, w, c) = img.dim();

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let rad = angle.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let mut output = Array3::<u8>::zeros((h, w, c));

    for i in 0..h {
        for j in 0..w {
            // Rotate around center
            let x = j as f32 - cx;
            let y = i as f32 - cy;

            let src_x = x * cos_a + y * sin_a + cx;
            let src_y = -x * sin_a + y * cos_a + cy;

            if src_x >= 0.0 && src_x < w as f32 - 1.0 && src_y >= 0.0 && src_y < h as f32 - 1.0 {
                // Bilinear interpolation
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let xf = src_x - x0 as f32;
                let yf = src_y - y0 as f32;

                for k in 0..c {
                    let v00 = img[[y0, x0, k]] as f32;
                    let v01 = img[[y0, x1, k]] as f32;
                    let v10 = img[[y1, x0, k]] as f32;
                    let v11 = img[[y1, x1, k]] as f32;

                    let val = v00 * (1.0 - xf) * (1.0 - yf)
                        + v01 * xf * (1.0 - yf)
                        + v10 * (1.0 - xf) * yf
                        + v11 * xf * yf;

                    output[[i, j, k]] = val.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    if let Some(path) = output_file_name {
        save_array(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

fn load_input(filename: Option<&str>, img: Option<PyReadonlyArray3<u8>>) -> PyResult<Array3<u8>> {
    if let Some(path) = filename {
        let dyn_img = image::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let rgb = dyn_img.to_rgb8();
        let (w, h) = rgb.dimensions();

        let mut arr = Array3::<u8>::zeros((h as usize, w as usize, 3));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            // RGB to BGR
            arr[[y as usize, x as usize, 0]] = pixel[2];
            arr[[y as usize, x as usize, 1]] = pixel[1];
            arr[[y as usize, x as usize, 2]] = pixel[0];
        }
        Ok(arr)
    } else if let Some(arr) = img {
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
        Ok(owned)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Either input_filename or input_img must be provided",
        ))
    }
}

fn resize_array(img: &Array3<u8>, new_h: usize, new_w: usize) -> Array3<u8> {
    let (h, w, c) = img.dim();
    let mut output = Array3::<u8>::zeros((new_h, new_w, c));

    let scale_y = h as f32 / new_h as f32;
    let scale_x = w as f32 / new_w as f32;

    for i in 0..new_h {
        for j in 0..new_w {
            let src_y = (i as f32 * scale_y).min((h - 1) as f32);
            let src_x = (j as f32 * scale_x).min((w - 1) as f32);

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let yf = src_y - y0 as f32;
            let xf = src_x - x0 as f32;

            for k in 0..c {
                let v00 = img[[y0, x0, k]] as f32;
                let v01 = img[[y0, x1, k]] as f32;
                let v10 = img[[y1, x0, k]] as f32;
                let v11 = img[[y1, x1, k]] as f32;

                let val = v00 * (1.0 - xf) * (1.0 - yf)
                    + v01 * xf * (1.0 - yf)
                    + v10 * (1.0 - xf) * yf
                    + v11 * xf * yf;

                output[[i, j, k]] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }

    output
}

fn save_array(arr: &Array3<u8>, path: &str) -> PyResult<()> {
    let (h, w, _c) = arr.dim();
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            // BGR to RGB
            img.put_pixel(
                x as u32,
                y as u32,
                Rgb([arr[[y, x, 2]], arr[[y, x, 1]], arr[[y, x, 0]]]),
            );
        }
    }

    img.save(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}
