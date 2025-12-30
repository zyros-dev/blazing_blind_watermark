use image::{ImageBuffer, Rgb};
use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

/// Estimate crop parameters using template matching
#[pyfunction]
#[pyo3(signature = (original_file=None, template_file=None, ori_img=None, tem_img=None, scale=(0.5, 2.0), search_num=200))]
pub fn estimate_crop_parameters(
    original_file: Option<&str>,
    template_file: Option<&str>,
    ori_img: Option<PyReadonlyArray3<u8>>,
    tem_img: Option<PyReadonlyArray3<u8>>,
    scale: (f32, f32),
    search_num: usize,
) -> PyResult<((usize, usize, usize, usize), (usize, usize), f32, f32)> {
    let ori_gray = load_grayscale(original_file, ori_img)?;
    let tem_gray = load_grayscale(template_file, tem_img)?;

    let (ori_h, ori_w) = (ori_gray.len(), ori_gray[0].len());
    let (tem_h, tem_w) = (tem_gray.len(), tem_gray[0].len());

    let (min_scale, max_scale) = scale;
    let max_scale = max_scale
        .min(ori_h as f32 / tem_h as f32)
        .min(ori_w as f32 / tem_w as f32);

    if scale.0 == scale.1 && (scale.0 - 1.0).abs() < 0.001 {
        let (best_y, best_x, best_score) = template_match(&ori_gray, &tem_gray);
        let loc = (best_x, best_y, best_x + tem_w, best_y + tem_h);
        return Ok((loc, (ori_h, ori_w), best_score, 1.0));
    }

    let mut best_result = ((0, 0, 0, 0), 0.0f32, 1.0f32);

    for pass in 0..2 {
        let (search_min, search_max, num) = if pass == 0 {
            (min_scale, max_scale, search_num)
        } else {
            let delta = (max_scale - min_scale) / search_num as f32;
            let refined_min = (best_result.2 - delta * 2.0).max(min_scale);
            let refined_max = (best_result.2 + delta * 2.0).min(max_scale);
            let refined_num =
                ((refined_max - refined_min) * tem_h.max(tem_w) as f32 * 2.0) as usize + 1;
            (refined_min, refined_max, refined_num.max(10))
        };

        for i in 0..num {
            let s = search_min + (search_max - search_min) * i as f32 / (num - 1).max(1) as f32;

            let scaled_h = (tem_h as f32 * s) as usize;
            let scaled_w = (tem_w as f32 * s) as usize;

            if scaled_h > ori_h || scaled_w > ori_w || scaled_h < 2 || scaled_w < 2 {
                continue;
            }

            let scaled = resize_gray(&tem_gray, scaled_h, scaled_w);
            let (best_y, best_x, score) = template_match(&ori_gray, &scaled);

            if score > best_result.1 {
                best_result = (
                    (best_x, best_y, best_x + scaled_w, best_y + scaled_h),
                    score,
                    s,
                );
            }
        }
    }

    Ok((best_result.0, (ori_h, ori_w), best_result.1, best_result.2))
}

/// Recover image from crop attack
#[pyfunction]
#[pyo3(signature = (template_file=None, tem_img=None, output_file_name=None, loc=None, image_o_shape=None))]
pub fn recover_crop<'py>(
    py: Python<'py>,
    template_file: Option<&str>,
    tem_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    loc: Option<(usize, usize, usize, usize)>,
    image_o_shape: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img = load_color(template_file, tem_img)?;
    let (_tem_h, _tem_w, c) = img.dim();

    let (x1, y1, x2, y2) =
        loc.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("loc is required"))?;

    let (ori_h, ori_w) = image_o_shape
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image_o_shape is required"))?;

    let mut output = Array3::<u8>::zeros((ori_h, ori_w, c));
    let target_h = y2 - y1;
    let target_w = x2 - x1;
    let resized = resize_color(&img, target_h, target_w);

    for i in 0..target_h.min(ori_h - y1) {
        for j in 0..target_w.min(ori_w - x1) {
            for k in 0..c {
                output[[y1 + i, x1 + j, k]] = resized[[i, j, k]];
            }
        }
    }

    if let Some(path) = output_file_name {
        save_color(&output, path)?;
    }

    Ok(output.to_pyarray_bound(py).to_owned())
}

fn load_grayscale(
    filename: Option<&str>,
    img: Option<PyReadonlyArray3<u8>>,
) -> PyResult<Vec<Vec<f32>>> {
    if let Some(path) = filename {
        let dyn_img = image::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let gray = dyn_img.to_luma8();
        let (w, h) = gray.dimensions();

        let mut result = vec![vec![0.0f32; w as usize]; h as usize];
        for y in 0..h {
            for x in 0..w {
                result[y as usize][x as usize] = gray.get_pixel(x, y).0[0] as f32;
            }
        }
        Ok(result)
    } else if let Some(arr) = img {
        let a = arr.as_array();
        let shape = a.shape();
        let h = shape[0];
        let w = shape[1];

        let mut result = vec![vec![0.0f32; w]; h];
        for i in 0..h {
            for j in 0..w {
                // Convert BGR to grayscale
                let b = a[[i, j, 0]] as f32;
                let g = a[[i, j, 1]] as f32;
                let r = a[[i, j, 2]] as f32;
                result[i][j] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }
        Ok(result)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Either filename or img must be provided",
        ))
    }
}

fn load_color(filename: Option<&str>, img: Option<PyReadonlyArray3<u8>>) -> PyResult<Array3<u8>> {
    if let Some(path) = filename {
        let dyn_img = image::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let rgb = dyn_img.to_rgb8();
        let (w, h) = rgb.dimensions();

        let mut arr = Array3::<u8>::zeros((h as usize, w as usize, 3));
        for (x, y, pixel) in rgb.enumerate_pixels() {
            arr[[y as usize, x as usize, 0]] = pixel[2]; // BGR
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
            "Either filename or img must be provided",
        ))
    }
}

fn resize_gray(img: &[Vec<f32>], new_h: usize, new_w: usize) -> Vec<Vec<f32>> {
    let h = img.len();
    let w = img[0].len();

    let scale_y = h as f32 / new_h as f32;
    let scale_x = w as f32 / new_w as f32;

    let mut output = vec![vec![0.0f32; new_w]; new_h];

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

            let v00 = img[y0][x0];
            let v01 = img[y0][x1];
            let v10 = img[y1][x0];
            let v11 = img[y1][x1];

            output[i][j] = v00 * (1.0 - xf) * (1.0 - yf)
                + v01 * xf * (1.0 - yf)
                + v10 * (1.0 - xf) * yf
                + v11 * xf * yf;
        }
    }

    output
}

fn resize_color(img: &Array3<u8>, new_h: usize, new_w: usize) -> Array3<u8> {
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

/// Template matching using normalized cross-correlation
fn template_match(image: &[Vec<f32>], template: &[Vec<f32>]) -> (usize, usize, f32) {
    let img_h = image.len();
    let img_w = image[0].len();
    let tem_h = template.len();
    let tem_w = template[0].len();

    if tem_h > img_h || tem_w > img_w {
        return (0, 0, 0.0);
    }

    let tem_mean: f32 = template.iter().flatten().sum::<f32>() / (tem_h * tem_w) as f32;
    let tem_std: f32 = (template
        .iter()
        .flatten()
        .map(|&x| (x - tem_mean).powi(2))
        .sum::<f32>()
        / (tem_h * tem_w) as f32)
        .sqrt();

    if tem_std < 1e-6 {
        return (0, 0, 0.0);
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_y = 0;
    let mut best_x = 0;

    for y in 0..=(img_h - tem_h) {
        for x in 0..=(img_w - tem_w) {
            let mut patch_sum = 0.0f32;
            for i in 0..tem_h {
                for j in 0..tem_w {
                    patch_sum += image[y + i][x + j];
                }
            }
            let patch_mean = patch_sum / (tem_h * tem_w) as f32;

            let mut numer = 0.0f32;
            let mut patch_var = 0.0f32;

            for i in 0..tem_h {
                for j in 0..tem_w {
                    let img_val = image[y + i][x + j] - patch_mean;
                    let tem_val = template[i][j] - tem_mean;
                    numer += img_val * tem_val;
                    patch_var += img_val.powi(2);
                }
            }

            let patch_std = (patch_var / (tem_h * tem_w) as f32).sqrt();
            if patch_std < 1e-6 {
                continue;
            }

            let score = numer / ((tem_h * tem_w) as f32 * patch_std * tem_std);

            if score > best_score {
                best_score = score;
                best_y = y;
                best_x = x;
            }
        }
    }

    (best_y, best_x, best_score)
}

fn save_color(arr: &Array3<u8>, path: &str) -> PyResult<()> {
    let (h, w, _) = arr.dim();
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
