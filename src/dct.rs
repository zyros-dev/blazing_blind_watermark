use std::f32::consts::PI;

/// Get DCT coefficient - computed at runtime for accuracy
/// cos(PI/N * (n + 0.5) * k) where N=4
#[inline]
fn dct_coeff(k: usize, n: usize) -> f32 {
    (PI / 4.0 * (n as f32 + 0.5) * k as f32).cos()
}

/// 1D DCT-II (scipy/opencv compatible)
fn dct_1d(input: &[f32; 4]) -> [f32; 4] {
    let mut output = [0.0f32; 4];
    let scale = (2.0f32 / 4.0).sqrt();

    for k in 0..4 {
        let mut sum = 0.0f32;
        for n in 0..4 {
            sum += input[n] * dct_coeff(k, n);
        }
        // Apply normalization: sqrt(1/N) for k=0, sqrt(2/N) otherwise
        output[k] = if k == 0 {
            sum * (1.0f32 / 4.0).sqrt()
        } else {
            sum * scale
        };
    }
    output
}

/// 1D IDCT (inverse) - orthonormal
fn idct_1d(input: &[f32; 4]) -> [f32; 4] {
    let mut output = [0.0f32; 4];
    let scale0 = (1.0f32 / 4.0).sqrt();
    let scale = (2.0f32 / 4.0).sqrt();

    for n in 0..4 {
        let mut sum = input[0] * scale0 * dct_coeff(0, n); // k=0 term (cos(0) = 1)
        for k in 1..4 {
            sum += input[k] * scale * dct_coeff(k, n);
        }
        output[n] = sum;
    }
    output
}

/// 2D DCT-II for a 4x4 block (scipy compatible)
pub fn dct2_4x4(block: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // First pass: DCT on rows
    let mut temp = [[0.0f32; 4]; 4];
    for i in 0..4 {
        temp[i] = dct_1d(&block[i]);
    }

    // Second pass: DCT on columns
    let mut result = [[0.0f32; 4]; 4];
    for j in 0..4 {
        let col = [temp[0][j], temp[1][j], temp[2][j], temp[3][j]];
        let dct_col = dct_1d(&col);
        for i in 0..4 {
            result[i][j] = dct_col[i];
        }
    }

    result
}

/// 2D IDCT (Inverse DCT) for a 4x4 block
pub fn idct2_4x4(block: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // First pass: IDCT on columns
    let mut temp = [[0.0f32; 4]; 4];
    for j in 0..4 {
        let col = [block[0][j], block[1][j], block[2][j], block[3][j]];
        let idct_col = idct_1d(&col);
        for i in 0..4 {
            temp[i][j] = idct_col[i];
        }
    }

    // Second pass: IDCT on rows
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        result[i] = idct_1d(&temp[i]);
    }

    result
}

/// Flatten a 4x4 block to a 16-element array
#[inline]
pub fn flatten_4x4(block: &[[f32; 4]; 4]) -> [f32; 16] {
    let mut flat = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            flat[i * 4 + j] = block[i][j];
        }
    }
    flat
}

/// Unflatten a 16-element array to a 4x4 block
#[inline]
pub fn unflatten_4x4(flat: &[f32; 16]) -> [[f32; 4]; 4] {
    let mut block = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            block[i][j] = flat[i * 4 + j];
        }
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        let block = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];

        let dct = dct2_4x4(&block);
        let reconstructed = idct2_4x4(&dct);

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (block[i][j] - reconstructed[i][j]).abs() < 0.01,
                    "Mismatch at [{i}][{j}]: expected {}, got {}",
                    block[i][j],
                    reconstructed[i][j]
                );
            }
        }
    }
}
