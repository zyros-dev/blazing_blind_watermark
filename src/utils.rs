use ndarray::Array3;
use rand_mt::Mt19937GenRand32;
use rayon::prelude::*;

/// Numpy-compatible random float generator (genrand_res53)
fn numpy_random_f64(rng: &mut Mt19937GenRand32) -> f64 {
    let a = (rng.next_u32() >> 5) as f64;
    let b = (rng.next_u32() >> 6) as f64;
    (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
}

/// Numpy-compatible bounded randint: returns value in [0, upper)
fn numpy_randint(rng: &mut Mt19937GenRand32, upper: u32) -> u32 {
    // Numpy uses rejection sampling to avoid modulo bias
    let mask = upper.next_power_of_two() - 1;
    loop {
        let val = rng.next_u32() & mask;
        if val < upper {
            return val;
        }
    }
}

/// Convert BGR to YUV color space (parallelized by row)
pub fn bgr_to_yuv(bgr: &Array3<f32>) -> Array3<f32> {
    let (h, w, _) = bgr.dim();
    let mut yuv_data = vec![0.0f32; h * w * 3];

    yuv_data
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(i, yuv_row)| {
            for j in 0..w {
                let b = bgr[[i, j, 0]];
                let g = bgr[[i, j, 1]];
                let r = bgr[[i, j, 2]];

                yuv_row[j * 3] = 0.299 * r + 0.587 * g + 0.114 * b;
                yuv_row[j * 3 + 1] = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
                yuv_row[j * 3 + 2] = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;
            }
        });

    Array3::from_shape_vec((h, w, 3), yuv_data).unwrap()
}

/// Convert YUV to BGR color space (parallelized by row)
pub fn yuv_to_bgr(yuv: &Array3<f32>) -> Array3<f32> {
    let (h, w, _) = yuv.dim();
    let mut bgr_data = vec![0.0f32; h * w * 3];

    bgr_data
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(i, bgr_row)| {
            for j in 0..w {
                let y = yuv[[i, j, 0]];
                let u = yuv[[i, j, 1]] - 128.0;
                let v = yuv[[i, j, 2]] - 128.0;

                bgr_row[j * 3] = y + 2.03211 * u; // B
                bgr_row[j * 3 + 1] = y - 0.39465 * u - 0.58060 * v; // G
                bgr_row[j * 3 + 2] = y + 1.13983 * v; // R
            }
        });

    Array3::from_shape_vec((h, w, 3), bgr_data).unwrap()
}

/// Generate shuffle indices for password-based encryption (pure Rust)
/// Matches: np.random.RandomState(seed).random(size=(size, block_shape)).argsort(axis=1)
pub fn generate_shuffle_indices(seed: u64, size: usize, block_len: usize) -> Vec<Vec<usize>> {
    let mut rng = Mt19937GenRand32::new(seed as u32);
    let mut result = Vec::with_capacity(size);

    for _ in 0..size {
        // Generate random values for this row
        let mut row: Vec<(f64, usize)> = (0..block_len)
            .map(|i| (numpy_random_f64(&mut rng), i))
            .collect();

        // Stable sort by random value (argsort)
        row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Extract indices
        result.push(row.into_iter().map(|(_, i)| i).collect());
    }

    result
}

/// Shuffle a flat array according to indices
#[inline]
pub fn shuffle_by_indices(data: &[f32; 16], indices: &[usize]) -> [f32; 16] {
    let mut shuffled = [0.0f32; 16];
    for (i, &idx) in indices.iter().enumerate() {
        shuffled[i] = data[idx];
    }
    shuffled
}

/// Unshuffle a flat array (inverse of shuffle)
#[inline]
pub fn unshuffle_by_indices(shuffled: &[f32; 16], indices: &[usize]) -> [f32; 16] {
    let mut data = [0.0f32; 16];
    for (i, &idx) in indices.iter().enumerate() {
        data[idx] = shuffled[i];
    }
    data
}

/// Shuffle watermark bits using password (pure Rust, numpy-compatible)
/// Matches: np.random.RandomState(password).shuffle(bits)
pub fn shuffle_wm_bits(bits: &[bool], password: u64) -> Vec<bool> {
    let mut result = bits.to_vec();
    let mut rng = Mt19937GenRand32::new(password as u32);

    // Fisher-Yates shuffle (numpy goes from n-1 down to 1)
    for i in (1..result.len()).rev() {
        let j = numpy_randint(&mut rng, (i + 1) as u32) as usize;
        result.swap(i, j);
    }

    result
}

/// Unshuffle watermark bits (for extraction, pure Rust)
pub fn unshuffle_wm_bits(shuffled: &[bool], password: u64, len: usize) -> Vec<bool> {
    // Generate the same shuffle permutation
    let mut indices: Vec<usize> = (0..len).collect();
    let mut rng = Mt19937GenRand32::new(password as u32);

    for i in (1..len).rev() {
        let j = numpy_randint(&mut rng, (i + 1) as u32) as usize;
        indices.swap(i, j);
    }

    // Invert the permutation
    let mut result = vec![false; len];
    for (i, &idx) in indices.iter().enumerate() {
        result[idx] = shuffled[i];
    }

    result
}

/// One-dimensional k-means clustering with 2 centers (for binary classification)
pub fn one_dim_kmeans(inputs: &[f32]) -> Vec<bool> {
    if inputs.is_empty() {
        return vec![];
    }

    let min_val = inputs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = inputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut center = [min_val, max_val];
    let e_tol = 1e-6;

    for _ in 0..300 {
        let threshold = (center[0] + center[1]) / 2.0;

        let mut sum0 = 0.0f32;
        let mut count0 = 0usize;
        let mut sum1 = 0.0f32;
        let mut count1 = 0usize;

        for &x in inputs {
            if x > threshold {
                sum1 += x;
                count1 += 1;
            } else {
                sum0 += x;
                count0 += 1;
            }
        }

        let new_center0 = if count0 > 0 { sum0 / count0 as f32 } else { center[0] };
        let new_center1 = if count1 > 0 { sum1 / count1 as f32 } else { center[1] };

        let new_threshold = (new_center0 + new_center1) / 2.0;
        if (new_threshold - threshold).abs() < e_tol {
            break;
        }

        center = [new_center0, new_center1];
    }

    let threshold = (center[0] + center[1]) / 2.0;
    inputs.iter().map(|&x| x > threshold).collect()
}

/// Pad image to even dimensions
pub fn pad_to_even(img: &Array3<f32>) -> Array3<f32> {
    let (h, w, c) = img.dim();
    let new_h = h + h % 2;
    let new_w = w + w % 2;

    if new_h == h && new_w == w {
        return img.clone();
    }

    let mut padded = Array3::<f32>::zeros((new_h, new_w, c));
    for i in 0..h {
        for j in 0..w {
            for k in 0..c {
                padded[[i, j, k]] = img[[i, j, k]];
            }
        }
    }
    padded
}

/// Convert string to bits (UTF-8 encoding, matching Python implementation)
pub fn string_to_bits(s: &str) -> Vec<bool> {
    let bytes = s.as_bytes();
    let hex_str = bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    let num = u128::from_str_radix(&hex_str, 16).unwrap_or(0);
    let binary = format!("{:b}", num);
    binary.chars().map(|c| c == '1').collect()
}

/// Convert bits to string (UTF-8 decoding)
pub fn bits_to_string(bits: &[bool]) -> String {
    if bits.is_empty() {
        return String::new();
    }

    let binary: String = bits.iter().map(|&b| if b { '1' } else { '0' }).collect();
    let padded_len = ((binary.len() + 7) / 8) * 8;
    let padded = format!("{:0>width$}", binary, width = padded_len);

    let bytes: Vec<u8> = (0..padded.len() / 8)
        .filter_map(|i| {
            let byte_str = &padded[i * 8..(i + 1) * 8];
            u8::from_str_radix(byte_str, 2).ok()
        })
        .collect();

    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_roundtrip() {
        let original = "Hello, World!";
        let bits = string_to_bits(original);
        let recovered = bits_to_string(&bits);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_kmeans() {
        let inputs = vec![0.1, 0.2, 0.15, 0.8, 0.9, 0.85];
        let result = one_dim_kmeans(&inputs);
        assert_eq!(result, vec![false, false, false, true, true, true]);
    }

    #[test]
    fn test_shuffle_unshuffle() {
        let original = vec![true, false, true, true, false, false, true, false];
        let shuffled = shuffle_wm_bits(&original, 42);
        let unshuffled = unshuffle_wm_bits(&shuffled, 42, original.len());
        assert_eq!(original, unshuffled);
    }
}
