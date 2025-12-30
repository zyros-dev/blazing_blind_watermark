use crate::dct::{dct2_4x4, flatten_4x4, idct2_4x4, unflatten_4x4};
use crate::dwt::{dwt2, idwt2};
use crate::utils::{
    bgr_to_yuv, generate_shuffle_indices, one_dim_kmeans, pad_to_even, shuffle_by_indices,
    unshuffle_by_indices, yuv_to_bgr,
};
use ndarray::{s, Array2, Array3};
use rayon::prelude::*;

const BLOCK_SIZE: usize = 4;
const D1: f32 = 36.0;
const D2: f32 = 20.0;

pub struct WaterMarkCore {
    pub password_img: u64,
    pub fast_mode: bool,
    img_shape: Option<(usize, usize)>,
    ca: [Option<Array2<f32>>; 3],
    hvd: [Option<(Array2<f32>, Array2<f32>, Array2<f32>)>; 3],
    ca_block_shape: Option<(usize, usize)>,
    block_num: usize,
    alpha: Option<Array2<u8>>,
}

impl WaterMarkCore {
    pub fn new(password_img: u64) -> Self {
        Self {
            password_img,
            fast_mode: false,
            img_shape: None,
            ca: [None, None, None],
            hvd: [None, None, None],
            ca_block_shape: None,
            block_num: 0,
            alpha: None,
        }
    }

    /// Read and preprocess image for embedding/extraction
    pub fn read_img_arr(&mut self, img: &Array3<u8>) {
        let (h, w, c) = img.dim();

        self.alpha = if c == 4 {
            Some(img.slice(s![.., .., 3]).mapv(|x| x).to_owned())
        } else {
            None
        };

        let channels = c.min(3);
        let mut img_f32 = Array3::<f32>::zeros((h, w, 3));
        for i in 0..h {
            for j in 0..w {
                for k in 0..channels {
                    img_f32[[i, j, k]] = img[[i, j, k]] as f32;
                }
            }
        }

        self.img_shape = Some((h, w));

        let img_padded = pad_to_even(&img_f32);
        let img_yuv = bgr_to_yuv(&img_padded);

        let (h_pad, w_pad, _) = img_yuv.dim();
        let ca_h = (h_pad + 1) / 2;
        let ca_w = (w_pad + 1) / 2;

        self.ca_block_shape = Some((ca_h / BLOCK_SIZE, ca_w / BLOCK_SIZE));
        self.block_num = (ca_h / BLOCK_SIZE) * (ca_w / BLOCK_SIZE);

        for channel in 0..3 {
            let channel_data = img_yuv.slice(s![.., .., channel]).to_owned();
            let (ca, hvd) = dwt2(&channel_data);
            self.ca[channel] = Some(ca);
            self.hvd[channel] = Some(hvd);
        }
    }

    /// Embed watermark bits into the image
    pub fn embed(&mut self, wm_bits: &[bool]) -> Array3<u8> {
        let wm_size = wm_bits.len();
        assert!(
            wm_size < self.block_num,
            "Watermark too large: {} bits, max {} blocks",
            wm_size,
            self.block_num
        );

        let (block_rows, block_cols) = self.ca_block_shape.unwrap();
        let shufflers = generate_shuffle_indices(self.password_img, self.block_num, 16);

        let mut embed_ca: [Option<Array2<f32>>; 3] = [None, None, None];

        for channel in 0..3 {
            let ca = self.ca[channel].as_ref().unwrap();
            let mut new_ca = ca.clone();
            let mut blocks: Vec<((usize, usize), [[f32; 4]; 4])> = Vec::with_capacity(self.block_num);

            for bi in 0..block_rows {
                for bj in 0..block_cols {
                    let mut block = [[0.0f32; 4]; 4];
                    for i in 0..4 {
                        for j in 0..4 {
                            block[i][j] = ca[[bi * 4 + i, bj * 4 + j]];
                        }
                    }
                    blocks.push(((bi, bj), block));
                }
            }

            let processed: Vec<_> = blocks
                .into_par_iter()
                .enumerate()
                .map(|(idx, ((bi, bj), block))| {
                    let wm_bit = wm_bits[idx % wm_size];
                    let shuffler = &shufflers[idx];
                    let processed = if self.fast_mode {
                        block_add_wm_fast(&block, wm_bit)
                    } else {
                        block_add_wm_slow(&block, shuffler, wm_bit)
                    };
                    ((bi, bj), processed)
                })
                .collect();

            for ((bi, bj), block) in processed {
                for i in 0..4 {
                    for j in 0..4 {
                        new_ca[[bi * 4 + i, bj * 4 + j]] = block[i][j];
                    }
                }
            }

            embed_ca[channel] = Some(new_ca);
        }

        let mut embed_yuv = Array3::<f32>::zeros((
            self.ca[0].as_ref().unwrap().dim().0 * 2,
            self.ca[0].as_ref().unwrap().dim().1 * 2,
            3,
        ));

        for channel in 0..3 {
            let ca = embed_ca[channel].as_ref().unwrap();
            let hvd = self.hvd[channel].as_ref().unwrap();
            let reconstructed = idwt2(ca, hvd);

            let (h, w) = reconstructed.dim();
            for i in 0..h {
                for j in 0..w {
                    embed_yuv[[i, j, channel]] = reconstructed[[i, j]];
                }
            }
        }

        let (orig_h, orig_w) = self.img_shape.unwrap();
        let embed_bgr = yuv_to_bgr(&embed_yuv);

        let out_channels = if self.alpha.is_some() { 4 } else { 3 };
        let mut output = Array3::<u8>::zeros((orig_h, orig_w, out_channels));

        for i in 0..orig_h {
            for j in 0..orig_w {
                for k in 0..3 {
                    output[[i, j, k]] = embed_bgr[[i, j, k]].clamp(0.0, 255.0) as u8;
                }
            }
        }

        if let Some(ref alpha) = self.alpha {
            for i in 0..orig_h {
                for j in 0..orig_w {
                    output[[i, j, 3]] = alpha[[i, j]];
                }
            }
        }

        output
    }

    /// Extract raw watermark values from all blocks
    pub fn extract_raw(&mut self, img: &Array3<u8>) -> Vec<Vec<f32>> {
        self.read_img_arr(img);

        let (block_rows, block_cols) = self.ca_block_shape.unwrap();
        let shufflers = generate_shuffle_indices(self.password_img, self.block_num, 16);

        let mut wm_block_bit = vec![vec![0.0f32; self.block_num]; 3];

        for channel in 0..3 {
            let ca = self.ca[channel].as_ref().unwrap();

            let blocks: Vec<[[f32; 4]; 4]> = (0..block_rows)
                .flat_map(|bi| {
                    (0..block_cols).map(move |bj| {
                        let mut block = [[0.0f32; 4]; 4];
                        for i in 0..4 {
                            for j in 0..4 {
                                block[i][j] = ca[[bi * 4 + i, bj * 4 + j]];
                            }
                        }
                        block
                    })
                })
                .collect();

            let results: Vec<f32> = blocks
                .par_iter()
                .enumerate()
                .map(|(idx, block)| {
                    let shuffler = &shufflers[idx];
                    if self.fast_mode {
                        block_get_wm_fast(block)
                    } else {
                        block_get_wm_slow(block, shuffler)
                    }
                })
                .collect();

            wm_block_bit[channel] = results;
        }

        wm_block_bit
    }

    /// Average extracted bits across channels and cyclic embedding
    pub fn extract_avg(&self, wm_block_bit: &[Vec<f32>], wm_size: usize) -> Vec<f32> {
        let mut wm_avg = vec![0.0f32; wm_size];

        for i in 0..wm_size {
            let mut sum = 0.0f32;
            let mut count = 0;

            for channel in 0..3 {
                let mut j = i;
                while j < wm_block_bit[channel].len() {
                    sum += wm_block_bit[channel][j];
                    count += 1;
                    j += wm_size;
                }
            }

            wm_avg[i] = if count > 0 { sum / count as f32 } else { 0.5 };
        }

        wm_avg
    }

    /// Extract watermark (returns averaged floating point values)
    pub fn extract(&mut self, img: &Array3<u8>, wm_size: usize) -> Vec<f32> {
        let wm_block_bit = self.extract_raw(img);
        self.extract_avg(&wm_block_bit, wm_size)
    }

    /// Extract watermark with k-means clustering (for str/bit modes)
    pub fn extract_with_kmeans(&mut self, img: &Array3<u8>, wm_size: usize) -> Vec<bool> {
        let wm_avg = self.extract(img, wm_size);
        one_dim_kmeans(&wm_avg)
    }
}

fn block_add_wm_slow(block: &[[f32; 4]; 4], shuffler: &[usize], wm_bit: bool) -> [[f32; 4]; 4] {
    let block_dct = dct2_4x4(block);
    let flat = flatten_4x4(&block_dct);
    let shuffled = shuffle_by_indices(&flat, shuffler);
    let shuffled_block = unflatten_4x4(&shuffled);

    let mat = faer::mat![
        [shuffled_block[0][0] as f64, shuffled_block[0][1] as f64, shuffled_block[0][2] as f64, shuffled_block[0][3] as f64],
        [shuffled_block[1][0] as f64, shuffled_block[1][1] as f64, shuffled_block[1][2] as f64, shuffled_block[1][3] as f64],
        [shuffled_block[2][0] as f64, shuffled_block[2][1] as f64, shuffled_block[2][2] as f64, shuffled_block[2][3] as f64],
        [shuffled_block[3][0] as f64, shuffled_block[3][1] as f64, shuffled_block[3][2] as f64, shuffled_block[3][3] as f64],
    ];

    let svd = mat.svd();
    let u = svd.u();
    let s_diag = svd.s_diagonal();
    let mut s_vals: [f64; 4] = [s_diag[0], s_diag[1], s_diag[2], s_diag[3]];
    let v = svd.v();

    let wm_val = if wm_bit { 1.0 } else { 0.0 };
    s_vals[0] = ((s_vals[0] / D1 as f64).floor() + 0.25 + 0.5 * wm_val) * D1 as f64;
    s_vals[1] = ((s_vals[1] / D2 as f64).floor() + 0.25 + 0.5 * wm_val) * D2 as f64;

    let s_diag = faer::mat![
        [s_vals[0], 0.0, 0.0, 0.0],
        [0.0, s_vals[1], 0.0, 0.0],
        [0.0, 0.0, s_vals[2], 0.0],
        [0.0, 0.0, 0.0, s_vals[3]],
    ];

    let reconstructed = u * s_diag * v.transpose();

    let mut recon_flat = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            recon_flat[i * 4 + j] = reconstructed[(i, j)] as f32;
        }
    }

    let unshuffled = unshuffle_by_indices(&recon_flat, shuffler);
    idct2_4x4(&unflatten_4x4(&unshuffled))
}

fn block_add_wm_fast(block: &[[f32; 4]; 4], wm_bit: bool) -> [[f32; 4]; 4] {
    let block_dct = dct2_4x4(block);

    let mat = faer::mat![
        [block_dct[0][0] as f64, block_dct[0][1] as f64, block_dct[0][2] as f64, block_dct[0][3] as f64],
        [block_dct[1][0] as f64, block_dct[1][1] as f64, block_dct[1][2] as f64, block_dct[1][3] as f64],
        [block_dct[2][0] as f64, block_dct[2][1] as f64, block_dct[2][2] as f64, block_dct[2][3] as f64],
        [block_dct[3][0] as f64, block_dct[3][1] as f64, block_dct[3][2] as f64, block_dct[3][3] as f64],
    ];

    let svd = mat.svd();
    let u = svd.u();
    let s_diag = svd.s_diagonal();
    let mut s_vals: [f64; 4] = [s_diag[0], s_diag[1], s_diag[2], s_diag[3]];
    let v = svd.v();

    let wm_val = if wm_bit { 1.0 } else { 0.0 };
    s_vals[0] = ((s_vals[0] / D1 as f64).floor() + 0.25 + 0.5 * wm_val) * D1 as f64;

    let s_diag = faer::mat![
        [s_vals[0], 0.0, 0.0, 0.0],
        [0.0, s_vals[1], 0.0, 0.0],
        [0.0, 0.0, s_vals[2], 0.0],
        [0.0, 0.0, 0.0, s_vals[3]],
    ];

    let reconstructed = u * s_diag * v.transpose();

    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = reconstructed[(i, j)] as f32;
        }
    }

    idct2_4x4(&result)
}

fn block_get_wm_slow(block: &[[f32; 4]; 4], shuffler: &[usize]) -> f32 {
    let block_dct = dct2_4x4(block);
    let flat = flatten_4x4(&block_dct);
    let shuffled = shuffle_by_indices(&flat, shuffler);
    let shuffled_block = unflatten_4x4(&shuffled);

    let mat = faer::mat![
        [shuffled_block[0][0] as f64, shuffled_block[0][1] as f64, shuffled_block[0][2] as f64, shuffled_block[0][3] as f64],
        [shuffled_block[1][0] as f64, shuffled_block[1][1] as f64, shuffled_block[1][2] as f64, shuffled_block[1][3] as f64],
        [shuffled_block[2][0] as f64, shuffled_block[2][1] as f64, shuffled_block[2][2] as f64, shuffled_block[2][3] as f64],
        [shuffled_block[3][0] as f64, shuffled_block[3][1] as f64, shuffled_block[3][2] as f64, shuffled_block[3][3] as f64],
    ];

    let svd = mat.svd();
    let s_diag = svd.s_diagonal();
    let s0 = s_diag[0];
    let s1 = s_diag[1];

    let wm0 = if s0 % D1 as f64 > D1 as f64 / 2.0 { 1.0 } else { 0.0 };
    let wm1 = if s1 % D2 as f64 > D2 as f64 / 2.0 { 1.0 } else { 0.0 };

    ((wm0 * 3.0 + wm1) / 4.0) as f32
}

fn block_get_wm_fast(block: &[[f32; 4]; 4]) -> f32 {
    let block_dct = dct2_4x4(block);

    let mat = faer::mat![
        [block_dct[0][0] as f64, block_dct[0][1] as f64, block_dct[0][2] as f64, block_dct[0][3] as f64],
        [block_dct[1][0] as f64, block_dct[1][1] as f64, block_dct[1][2] as f64, block_dct[1][3] as f64],
        [block_dct[2][0] as f64, block_dct[2][1] as f64, block_dct[2][2] as f64, block_dct[2][3] as f64],
        [block_dct[3][0] as f64, block_dct[3][1] as f64, block_dct[3][2] as f64, block_dct[3][3] as f64],
    ];

    let svd = mat.svd();
    let s0 = svd.s_diagonal()[0];

    if s0 % D1 as f64 > D1 as f64 / 2.0 {
        1.0
    } else {
        0.0
    }
}
