use ndarray::Array2;

/// 2D Haar Discrete Wavelet Transform (pywavelets compatible)
/// Returns (LL, (LH, HL, HH)) where LL is the approximation coefficients
pub fn dwt2(input: &Array2<f32>) -> (Array2<f32>, (Array2<f32>, Array2<f32>, Array2<f32>)) {
    let (h, w) = input.dim();
    let h2 = h / 2;
    let w2 = w / 2;

    let mut ll = Array2::<f32>::zeros((h2, w2));
    let mut lh = Array2::<f32>::zeros((h2, w2));
    let mut hl = Array2::<f32>::zeros((h2, w2));
    let mut hh = Array2::<f32>::zeros((h2, w2));

    for i in 0..h2 {
        for j in 0..w2 {
            let a = input[[2 * i, 2 * j]];
            let b = input[[2 * i, 2 * j + 1]];
            let c = input[[2 * i + 1, 2 * j]];
            let d = input[[2 * i + 1, 2 * j + 1]];

            // Haar wavelet coefficients (matching pywavelets)
            ll[[i, j]] = (a + b + c + d) * 0.5;
            lh[[i, j]] = (a + b - c - d) * 0.5;
            hl[[i, j]] = (a - b + c - d) * 0.5;
            hh[[i, j]] = (a - b - c + d) * 0.5;  // Note: +d not -d
        }
    }

    (ll, (lh, hl, hh))
}

/// Inverse 2D Haar Discrete Wavelet Transform
pub fn idwt2(
    ll: &Array2<f32>,
    details: &(Array2<f32>, Array2<f32>, Array2<f32>),
) -> Array2<f32> {
    let (lh, hl, hh) = details;
    let (h2, w2) = ll.dim();
    let h = h2 * 2;
    let w = w2 * 2;

    let mut output = Array2::<f32>::zeros((h, w));

    for i in 0..h2 {
        for j in 0..w2 {
            let ll_v = ll[[i, j]];
            let lh_v = lh[[i, j]];
            let hl_v = hl[[i, j]];
            let hh_v = hh[[i, j]];

            // Inverse Haar: reconstruct 2x2 block
            output[[2 * i, 2 * j]] = (ll_v + lh_v + hl_v + hh_v) * 0.5;
            output[[2 * i, 2 * j + 1]] = (ll_v + lh_v - hl_v - hh_v) * 0.5;
            output[[2 * i + 1, 2 * j]] = (ll_v - lh_v + hl_v - hh_v) * 0.5;
            output[[2 * i + 1, 2 * j + 1]] = (ll_v - lh_v - hl_v + hh_v) * 0.5;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dwt_idwt_roundtrip() {
        let input = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let (ll, details) = dwt2(&input);
        let reconstructed = idwt2(&ll, &details);

        for i in 0..4 {
            for j in 0..4 {
                assert!((input[[i, j]] - reconstructed[[i, j]]).abs() < 1e-5);
            }
        }
    }
}
