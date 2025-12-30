use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
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
    let cv2 = py.import_bound("cv2")?;

    let img = load_input_cv2(py, input_filename, input_img)?;
    let shape = img.getattr("shape")?;
    let shape_tuple: (usize, usize, usize) = shape.extract()?;
    let (h, w, _c) = shape_tuple;

    let (x1, y1, x2, y2) = loc.unwrap_or((0, 0, w, h));

    // Crop using numpy slicing via Python builtins
    let builtins = py.import_bound("builtins")?;
    let slice_fn = builtins.getattr("slice")?;
    let row_slice = slice_fn.call1((y1, y2))?;
    let col_slice = slice_fn.call1((x1, x2))?;
    let cropped = img.get_item((row_slice, col_slice))?;

    let output = if let Some(s) = scale {
        if (s - 1.0).abs() > 0.001 {
            let crop_h = y2 - y1;
            let crop_w = x2 - x1;
            let new_h = (crop_h as f32 * s) as i32;
            let new_w = (crop_w as f32 * s) as i32;
            let dsize = PyTuple::new_bound(py, [new_w, new_h]);
            cv2.call_method("resize", (&cropped, dsize), None)?
        } else {
            cropped
        }
    } else {
        cropped
    };

    if let Some(path) = output_file_name {
        cv2.call_method("imwrite", (path, &output), None)?;
    }

    // Convert to numpy array and return
    let np = py.import_bound("numpy")?;
    let result = np.call_method("ascontiguousarray", (&output,), None)?;
    let arr: PyReadonlyArray3<u8> = result.extract()?;
    let owned = arr_to_owned(&arr);
    Ok(owned.to_pyarray_bound(py).to_owned())
}

/// Resize attack - uses cv2.resize for exact compatibility
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, out_shape=(500, 500)))]
pub fn resize_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    out_shape: (usize, usize),
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let cv2 = py.import_bound("cv2")?;

    let img = load_input_cv2(py, input_filename, input_img)?;
    let dsize = PyTuple::new_bound(py, [out_shape.0 as i32, out_shape.1 as i32]);
    let resized = cv2.call_method("resize", (&img, dsize), None)?;

    if let Some(path) = output_file_name {
        cv2.call_method("imwrite", (path, &resized), None)?;
    }

    let np = py.import_bound("numpy")?;
    let result = np.call_method("ascontiguousarray", (&resized,), None)?;
    let arr: PyReadonlyArray3<u8> = result.extract()?;
    let owned = arr_to_owned(&arr);
    Ok(owned.to_pyarray_bound(py).to_owned())
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
    let cv2 = py.import_bound("cv2")?;
    let np = py.import_bound("numpy")?;

    let img = load_input_cv2(py, input_filename, input_img)?;

    // Match Python: (img * ratio).clip(0, 255).astype(np.uint8)
    let scaled = img.call_method("__mul__", (ratio,), None)?;
    let clipped = np.call_method("clip", (&scaled, 0, 255), None)?;
    let output = clipped.call_method("astype", (np.getattr("uint8")?,), None)?;

    if let Some(path) = output_file_name {
        cv2.call_method("imwrite", (path, &output), None)?;
    }

    let result = np.call_method("ascontiguousarray", (&output,), None)?;
    let arr: PyReadonlyArray3<u8> = result.extract()?;
    let owned = arr_to_owned(&arr);
    Ok(owned.to_pyarray_bound(py).to_owned())
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
    let cv2 = py.import_bound("cv2")?;
    let np = py.import_bound("numpy")?;

    let img = load_input_cv2(py, input_filename, input_img)?;
    let output = img.call_method("copy", (), None)?;

    let shape = output.getattr("shape")?;
    let shape_tuple: (usize, usize, usize) = shape.extract()?;
    let (h, w, _c) = shape_tuple;

    let mut rng = rand::thread_rng();
    let builtins = py.import_bound("builtins")?;
    let slice_fn = builtins.getattr("slice")?;

    for _ in 0..n {
        let tmp_h = rng.gen::<f32>() * (1.0 - ratio);
        let start_h = (tmp_h * h as f32) as usize;
        let end_h = ((tmp_h + ratio) * h as f32) as usize;

        let tmp_w = rng.gen::<f32>() * (1.0 - ratio);
        let start_w = (tmp_w * w as f32) as usize;
        let end_w = ((tmp_w + ratio) * w as f32) as usize;

        // Set region to white
        let row_slice = slice_fn.call1((start_h, end_h.min(h)))?;
        let col_slice = slice_fn.call1((start_w, end_w.min(w)))?;
        output.set_item((row_slice, col_slice), 255u8)?;
    }

    if let Some(path) = output_file_name {
        cv2.call_method("imwrite", (path, &output), None)?;
    }

    let result = np.call_method("ascontiguousarray", (&output,), None)?;
    let arr: PyReadonlyArray3<u8> = result.extract()?;
    let owned = arr_to_owned(&arr);
    Ok(owned.to_pyarray_bound(py).to_owned())
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
    let cv2 = py.import_bound("cv2")?;
    let np = py.import_bound("numpy")?;

    let img = load_input_cv2(py, input_filename, input_img)?;
    let output = img.call_method("copy", (), None)?;

    let shape = output.getattr("shape")?;
    let shape_tuple: (usize, usize, usize) = shape.extract()?;
    let (h, w, _c) = shape_tuple;

    let mut rng = rand::thread_rng();

    for i in 0..h {
        for j in 0..w {
            if rng.gen::<f32>() < ratio {
                let idx = PyTuple::new_bound(py, [i, j]);
                output.set_item(idx, 255u8)?;
            }
        }
    }

    if let Some(path) = output_file_name {
        cv2.call_method("imwrite", (path, &output), None)?;
    }

    let result = np.call_method("ascontiguousarray", (&output,), None)?;
    let arr: PyReadonlyArray3<u8> = result.extract()?;
    let owned = arr_to_owned(&arr);
    Ok(owned.to_pyarray_bound(py).to_owned())
}

/// Rotation attack - uses cv2.warpAffine for exact compatibility
#[pyfunction]
#[pyo3(signature = (input_filename=None, input_img=None, output_file_name=None, angle=45.0))]
pub fn rot_att<'py>(
    py: Python<'py>,
    input_filename: Option<&str>,
    input_img: Option<PyReadonlyArray3<u8>>,
    output_file_name: Option<&str>,
    angle: f32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let cv2 = py.import_bound("cv2")?;
    let np = py.import_bound("numpy")?;

    let img = load_input_cv2(py, input_filename, input_img)?;

    let shape = img.getattr("shape")?;
    let shape_tuple: (usize, usize, usize) = shape.extract()?;
    let (rows, cols, _c) = shape_tuple;

    let center = PyTuple::new_bound(py, [cols as f32 / 2.0, rows as f32 / 2.0]);
    let rot_mat = cv2.call_method("getRotationMatrix2D", (center, angle, 1.0), None)?;

    let dsize = PyTuple::new_bound(py, [cols as i32, rows as i32]);
    let rotated = cv2.call_method("warpAffine", (&img, &rot_mat, dsize), None)?;

    if let Some(path) = output_file_name {
        cv2.call_method("imwrite", (path, &rotated), None)?;
    }

    let result = np.call_method("ascontiguousarray", (&rotated,), None)?;
    let arr: PyReadonlyArray3<u8> = result.extract()?;
    let owned = arr_to_owned(&arr);
    Ok(owned.to_pyarray_bound(py).to_owned())
}

fn load_input_cv2<'py>(
    py: Python<'py>,
    filename: Option<&str>,
    img: Option<PyReadonlyArray3<u8>>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let cv2 = py.import_bound("cv2")?;
    let np = py.import_bound("numpy")?;

    if let Some(path) = filename {
        let result = cv2.call_method("imread", (path,), None)?;
        Ok(result)
    } else if let Some(arr) = img {
        // Convert to contiguous array
        let py_arr = arr.as_array();
        let owned = arr_to_owned_from_view(&py_arr);
        let np_arr = owned.to_pyarray_bound(py);
        let result = np.call_method("ascontiguousarray", (&np_arr,), None)?;
        Ok(result)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Either input_filename or input_img must be provided",
        ))
    }
}

fn arr_to_owned(arr: &PyReadonlyArray3<u8>) -> Array3<u8> {
    let a = arr.as_array();
    arr_to_owned_from_view(&a)
}

fn arr_to_owned_from_view(a: &ndarray::ArrayView3<u8>) -> Array3<u8> {
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
}
