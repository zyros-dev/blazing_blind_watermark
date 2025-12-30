#![deny(dead_code, unused_imports, unused_variables)]

mod attack;
mod core;
mod dct;
mod dwt;
mod recover;
mod utils;
mod watermark;

use pyo3::prelude::*;

/// Recommended embed_size for fast mode. Balances speed and quality.
const RECOMMENDED_EMBED_SIZE: u32 = 1536;

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RECOMMENDED_EMBED_SIZE", RECOMMENDED_EMBED_SIZE)?;
    m.add_class::<watermark::WaterMark>()?;

    // Attack functions
    m.add_function(wrap_pyfunction!(attack::cut_att3, m)?)?;
    m.add_function(wrap_pyfunction!(attack::resize_att, m)?)?;
    m.add_function(wrap_pyfunction!(attack::bright_att, m)?)?;
    m.add_function(wrap_pyfunction!(attack::shelter_att, m)?)?;
    m.add_function(wrap_pyfunction!(attack::salt_pepper_att, m)?)?;
    m.add_function(wrap_pyfunction!(attack::rot_att, m)?)?;

    // Recovery functions
    m.add_function(wrap_pyfunction!(recover::estimate_crop_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(recover::recover_crop, m)?)?;

    Ok(())
}
