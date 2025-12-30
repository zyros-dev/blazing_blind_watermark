// Precomputed DCT-II coefficients: cos(PI/N * (n + 0.5) * k) where N=4
// DCT[k][n] for k=0..3, n=0..3
const DCT: [[f32; 4]; 4] = [
    [1.0, 1.0, 1.0, 1.0], // k=0: all cos(0) = 1
    [0.9238795325, 0.3826834324, -0.3826834324, -0.9238795325], // k=1: cos(π/8), cos(3π/8), cos(5π/8), cos(7π/8)
    [0.7071067812, -0.7071067812, -0.7071067812, 0.7071067812], // k=2: cos(π/4), cos(3π/4), cos(5π/4), cos(7π/4)
    [0.3826834324, -0.9238795325, 0.9238795325, -0.3826834324], // k=3: cos(3π/8), cos(9π/8), cos(15π/8), cos(21π/8)
];

// Normalization constants
const SCALE0: f32 = 0.5; // sqrt(1/4)
const SCALE: f32 = 0.707106781; // sqrt(2/4) = sqrt(0.5)

/// 1D DCT-II with precomputed coefficients
#[inline]
fn dct_1d(input: &[f32; 4]) -> [f32; 4] {
    [
        SCALE0 * (input[0] * DCT[0][0] + input[1] * DCT[0][1] + input[2] * DCT[0][2] + input[3] * DCT[0][3]),
        SCALE * (input[0] * DCT[1][0] + input[1] * DCT[1][1] + input[2] * DCT[1][2] + input[3] * DCT[1][3]),
        SCALE * (input[0] * DCT[2][0] + input[1] * DCT[2][1] + input[2] * DCT[2][2] + input[3] * DCT[2][3]),
        SCALE * (input[0] * DCT[3][0] + input[1] * DCT[3][1] + input[2] * DCT[3][2] + input[3] * DCT[3][3]),
    ]
}

/// 1D IDCT with precomputed coefficients
#[inline]
fn idct_1d(input: &[f32; 4]) -> [f32; 4] {
    let s0 = input[0] * SCALE0;
    let s1 = input[1] * SCALE;
    let s2 = input[2] * SCALE;
    let s3 = input[3] * SCALE;
    [
        s0 * DCT[0][0] + s1 * DCT[1][0] + s2 * DCT[2][0] + s3 * DCT[3][0],
        s0 * DCT[0][1] + s1 * DCT[1][1] + s2 * DCT[2][1] + s3 * DCT[3][1],
        s0 * DCT[0][2] + s1 * DCT[1][2] + s2 * DCT[2][2] + s3 * DCT[3][2],
        s0 * DCT[0][3] + s1 * DCT[1][3] + s2 * DCT[2][3] + s3 * DCT[3][3],
    ]
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
