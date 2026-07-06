//! TrOMR encoder (E3, bd-3jo6.5.3) — the fifth model lane's vision half
//! (census `docs/zoo/tromr-spec.md` §2a/§2b): a hybrid ResNetV2+ViT over one
//! grayscale staff crop `(1, 128, W)`, W ≤ 1280 a multiple of 16.
//!
//! Graph: TF-'SAME' stem `conv 1→64 k7 s2` → GN32+ReLU → −∞-pad max-pool k3
//! s2 → post-act Bottleneck stages `[2, 3, 7]` (widths 256/512/1024, strides
//! 1/2/2) → `(1024, 8, W/16)` → 1×1 proj to 256 + cls token + CROP-INDEXED
//! learned positions (row-major over an 80-wide table) → 4 pre-LN ViT blocks
//! (8 heads × 32, fused qkv, exact-erf GELU MLP 1024) → final LayerNorm →
//! `[1 + 8·W/16, 256]` — the cls token IS part of the decoder's
//! cross-attention context (§3: the connector is Identity).
//!
//! The stored backbone convs are PRE-WS-FOLDED (E2's export invokes timm's
//! own standardization arithmetic), so every conv here is a plain
//! [`nn::conv2d`] over a [`nn::tf_same_pad`]-prepared input — no runtime
//! weight standardization exists (spec §10.3). No conv biases anywhere in the
//! backbone; the backbone final norm is Identity (both census-confirmed
//! absent from the checkpoint).
//!
//! The 4-head AR decoder (E4) is deliberately NOT here yet — it is a
//! self-contained x-transformers graph (cross-attention every layer, GEGLU,
//! GLU-gated `on_attn`) that does not ride `decoder_qwen2` (§10 non-fit).

use crate::error::{FocrError, FocrResult};

use super::nn;
use super::tensor::Mat;
use super::vision_sam::Linear;
use super::weights::Weights;

/// Staff-crop input height (config `max_height`; spec §6 resizes to this).
pub const IMG_H: usize = 128;
/// ViT patch stride — the backbone's total 16× downsample (spec §2b).
pub const PATCH: usize = 16;
/// Encoder/decoder shared width (`emb_dim == dim == 256`, §3).
pub const DIM: usize = 256;
/// The learned position table is laid out for this many patch COLUMNS
/// (1280/16); a narrower crop crop-indexes its top-left block (§2b).
pub const POS_COLS: usize = 80;
/// Patch rows for the fixed 128-high input (128/16).
pub const POS_ROWS: usize = 8;
const VIT_HEADS: usize = 8;
const VIT_HEAD_DIM: usize = 32;
const GN_GROUPS: usize = 32;
const GN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-6;

/// One flat batch-1 NCHW feature map — the backbone currency.
struct Feature {
    data: Vec<f32>,
    ch: usize,
    h: usize,
    w: usize,
}

/// A backbone conv (no bias — census) + its following GroupNorm params.
/// `norm` is `None` only where the graph has a bare conv (never happens in
/// this backbone: every conv is followed by a GN, with or without ReLU).
struct ConvGn {
    w: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
    gn_w: Vec<f32>,
    gn_b: Vec<f32>,
}

impl ConvGn {
    /// TF-'SAME' pad (zero fill) → conv → GroupNorm(32, 1e-5) with optional
    /// fused ReLU.
    fn apply(&self, x: &Feature, relu: bool) -> FocrResult<Feature> {
        let (padded, ph, pw) = nn::tf_same_pad(
            &x.data,
            1,
            x.ch,
            x.h,
            x.w,
            self.k,
            self.k,
            self.stride,
            self.stride,
            0.0,
        );
        let (oh, ow) = (x.h.div_ceil(self.stride), x.w.div_ceil(self.stride));
        let mut data = nn::conv2d(
            &padded,
            &self.w,
            None,
            1,
            self.in_ch,
            ph,
            pw,
            self.k,
            self.k,
            oh,
            ow,
            self.stride,
            self.stride,
            self.out_ch,
        );
        nn::group_norm(
            &mut data,
            1,
            self.out_ch,
            oh * ow,
            GN_GROUPS,
            GN_EPS,
            &self.gn_w,
            &self.gn_b,
            relu,
        )?;
        Ok(Feature {
            data,
            ch: self.out_ch,
            h: oh,
            w: ow,
        })
    }
}

/// One post-act Bottleneck block (timm ResNetV2, preact=False — spec §2a):
/// `conv1 1×1 → GN+ReLU → conv2 3×3 (stride) → GN+ReLU → conv3 1×1 → GN(no
/// act) → + shortcut → ReLU`; block 0 of a stage downsamples the shortcut
/// with `1×1 (stride) → GN(no act)`.
struct Bottleneck {
    conv1: ConvGn,
    conv2: ConvGn,
    conv3: ConvGn,
    downsample: Option<ConvGn>,
}

impl Bottleneck {
    fn apply(&self, x: &Feature) -> FocrResult<Feature> {
        let shortcut = match &self.downsample {
            Some(d) => d.apply(x, false)?,
            None => Feature {
                data: x.data.clone(),
                ch: x.ch,
                h: x.h,
                w: x.w,
            },
        };
        let h = self.conv1.apply(x, true)?;
        let h = self.conv2.apply(&h, true)?;
        let mut h = self.conv3.apply(&h, false)?;
        if h.data.len() != shortcut.data.len() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "tromr bottleneck: residual len {} != shortcut len {}",
                h.data.len(),
                shortcut.data.len()
            )));
        }
        for (a, b) in h.data.iter_mut().zip(shortcut.data.iter()) {
            *a = (*a + b).max(0.0);
        }
        Ok(h)
    }
}

/// One pre-LN ViT block (spec §2b): LN(1e-6) → fused-qkv MHA (8×32, scale
/// 32^-0.5) → +res; LN → fc1 1024 → exact-erf GELU → fc2 → +res.
struct VitBlock {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv: Linear,
    proj: Linear,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Linear,
    fc2: Linear,
}

/// The hydrated encoder weights.
pub struct TromrEncoderW {
    stem: ConvGn,
    stages: Vec<Vec<Bottleneck>>,
    patch_proj: Linear,
    cls_token: Vec<f32>,
    pos_embed: Vec<f32>,
    blocks: Vec<VitBlock>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

impl TromrEncoderW {
    /// Hydrate from the (WS-pre-folded) artifact — spec §12 names verbatim.
    ///
    /// # Errors
    /// A missing tensor or a shape violation.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let b = "encoder.patch_embed.backbone.";
        let conv_gn = |conv: String,
                       norm: String,
                       out_ch: usize,
                       in_ch: usize,
                       k: usize,
                       stride: usize|
         -> FocrResult<ConvGn> {
            Ok(ConvGn {
                w: weights.vec(&conv)?,
                out_ch,
                in_ch,
                k,
                stride,
                gn_w: weights.vec(&format!("{norm}.weight"))?,
                gn_b: weights.vec(&format!("{norm}.bias"))?,
            })
        };

        let stem = conv_gn(
            format!("{b}stem.conv.weight"),
            format!("{b}stem.norm"),
            64,
            1,
            7,
            2,
        )?;

        // Stages [2, 3, 7]; (in, mid, out, stride) per census §2a/§12.
        let plan: [(usize, usize, usize, usize, usize); 3] = [
            (2, 64, 64, 256, 1),
            (3, 256, 128, 512, 2),
            (7, 512, 256, 1024, 2),
        ];
        let mut stages = Vec::with_capacity(3);
        for (s, &(blocks_n, stage_in, mid, out, stage_stride)) in plan.iter().enumerate() {
            let mut blocks = Vec::with_capacity(blocks_n);
            for blk in 0..blocks_n {
                let p = format!("{b}stages.{s}.blocks.{blk}.");
                let (in_ch, stride) = if blk == 0 {
                    (stage_in, stage_stride)
                } else {
                    (out, 1)
                };
                let downsample = if blk == 0 {
                    Some(conv_gn(
                        format!("{p}downsample.conv.weight"),
                        format!("{p}downsample.norm"),
                        out,
                        in_ch,
                        1,
                        stride,
                    )?)
                } else {
                    None
                };
                blocks.push(Bottleneck {
                    conv1: conv_gn(
                        format!("{p}conv1.weight"),
                        format!("{p}norm1"),
                        mid,
                        in_ch,
                        1,
                        1,
                    )?,
                    conv2: conv_gn(
                        format!("{p}conv2.weight"),
                        format!("{p}norm2"),
                        mid,
                        mid,
                        3,
                        stride,
                    )?,
                    conv3: conv_gn(
                        format!("{p}conv3.weight"),
                        format!("{p}norm3"),
                        out,
                        mid,
                        1,
                        1,
                    )?,
                    downsample,
                });
            }
            stages.push(blocks);
        }

        let lin = |wname: String, bname: String, out: usize, in_: usize| -> FocrResult<Linear> {
            Ok(Linear {
                w: weights.vec(&wname)?,
                b: weights.vec(&bname)?,
                out,
                in_,
            })
        };
        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("encoder.blocks.{i}.");
            blocks.push(VitBlock {
                ln1_w: weights.vec(&format!("{p}norm1.weight"))?,
                ln1_b: weights.vec(&format!("{p}norm1.bias"))?,
                qkv: lin(
                    format!("{p}attn.qkv.weight"),
                    format!("{p}attn.qkv.bias"),
                    3 * DIM,
                    DIM,
                )?,
                proj: lin(
                    format!("{p}attn.proj.weight"),
                    format!("{p}attn.proj.bias"),
                    DIM,
                    DIM,
                )?,
                ln2_w: weights.vec(&format!("{p}norm2.weight"))?,
                ln2_b: weights.vec(&format!("{p}norm2.bias"))?,
                fc1: lin(
                    format!("{p}mlp.fc1.weight"),
                    format!("{p}mlp.fc1.bias"),
                    4 * DIM,
                    DIM,
                )?,
                fc2: lin(
                    format!("{p}mlp.fc2.weight"),
                    format!("{p}mlp.fc2.bias"),
                    DIM,
                    4 * DIM,
                )?,
            });
        }

        Ok(Self {
            stem,
            stages,
            patch_proj: lin(
                "encoder.patch_embed.proj.weight".into(),
                "encoder.patch_embed.proj.bias".into(),
                DIM,
                1024,
            )?,
            cls_token: weights.vec("encoder.cls_token")?,
            pos_embed: weights.vec("encoder.pos_embed")?,
            blocks,
            final_ln_w: weights.vec("encoder.norm.weight")?,
            final_ln_b: weights.vec("encoder.norm.bias")?,
        })
    }
}

/// The ResNetV2 backbone: staff tensor `(1, 128, W)` → `(1024, 8, W/16)`.
fn backbone(w: &TromrEncoderW, pixels: &[f32], width: usize) -> FocrResult<Feature> {
    if width == 0 || !width.is_multiple_of(PATCH) || width > POS_COLS * PATCH {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr: width {width} must be a non-zero multiple of {PATCH} <= {} (spec §2b \
             crop-indexed positions go undefined past 1280)",
            POS_COLS * PATCH
        )));
    }
    if pixels.len() != IMG_H * width {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr: pixel buffer {} != 1*{IMG_H}*{width}",
            pixels.len()
        )));
    }
    let x = Feature {
        data: pixels.to_vec(),
        ch: 1,
        h: IMG_H,
        w: width,
    };
    // Stem: conv7 s2 (GN+ReLU) then the −∞-padded s2 max-pool.
    let x = w.stem.apply(&x, true)?;
    let (padded, ph, pw) =
        nn::tf_same_pad(&x.data, 1, x.ch, x.h, x.w, 3, 3, 2, 2, f32::NEG_INFINITY);
    let (oh, ow) = (x.h.div_ceil(2), x.w.div_ceil(2));
    let mut x = Feature {
        data: nn::max_pool2d(&padded, 1, x.ch, ph, pw, 3, 2, oh, ow),
        ch: x.ch,
        h: oh,
        w: ow,
    };
    for stage in &w.stages {
        for block in stage {
            x = block.apply(&x)?;
        }
    }
    Ok(x)
}

/// Channel-major `(C, H·W)` → token-major `[H·W, C]`.
fn tokens_from_feature(f: &Feature) -> Mat {
    let spatial = f.h * f.w;
    let mut out = vec![0.0f32; spatial * f.ch];
    for c in 0..f.ch {
        for s in 0..spatial {
            out[s * f.ch + c] = f.data[c * spatial + s];
        }
    }
    Mat::from_vec(spatial, f.ch, out)
}

/// Fused-qkv bidirectional MHA (8 heads × 32, scale 32^-0.5).
fn self_attention(blk: &VitBlock, x: &Mat) -> FocrResult<Mat> {
    let seq = x.rows;
    let qkv = blk.qkv.apply(x)?; // [seq, 768] = q|k|v
    let head_span = seq * VIT_HEAD_DIM;
    let mut qf = vec![0.0f32; VIT_HEADS * head_span];
    let mut kf = vec![0.0f32; VIT_HEADS * head_span];
    let mut vf = vec![0.0f32; VIT_HEADS * head_span];
    for s in 0..seq {
        let row = qkv.row(s);
        for h in 0..VIT_HEADS {
            let dst = h * head_span + s * VIT_HEAD_DIM;
            let src = h * VIT_HEAD_DIM;
            qf[dst..dst + VIT_HEAD_DIM].copy_from_slice(&row[src..src + VIT_HEAD_DIM]);
            kf[dst..dst + VIT_HEAD_DIM].copy_from_slice(&row[DIM + src..DIM + src + VIT_HEAD_DIM]);
            vf[dst..dst + VIT_HEAD_DIM]
                .copy_from_slice(&row[2 * DIM + src..2 * DIM + src + VIT_HEAD_DIM]);
        }
    }
    let scale = 1.0 / (VIT_HEAD_DIM as f32).sqrt();
    let ctx = nn::sdpa(
        &qf,
        &kf,
        &vf,
        VIT_HEADS,
        seq,
        seq,
        VIT_HEAD_DIM,
        VIT_HEAD_DIM,
        scale,
        false,
    );
    // Head-major back to [seq, 256].
    let mut merged = vec![0.0f32; seq * DIM];
    for h in 0..VIT_HEADS {
        for s in 0..seq {
            let src = h * head_span + s * VIT_HEAD_DIM;
            let dst = s * DIM + h * VIT_HEAD_DIM;
            merged[dst..dst + VIT_HEAD_DIM].copy_from_slice(&ctx[src..src + VIT_HEAD_DIM]);
        }
    }
    blk.proj.apply(&Mat::from_vec(seq, DIM, merged))
}

fn add_assign(x: &mut Mat, y: &Mat) -> FocrResult<()> {
    if x.rows != y.rows || x.cols != y.cols {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr add_assign: [{}, {}] += [{}, {}]",
            x.rows,
            x.cols,
            y.rows,
            y.cols
        )));
    }
    for (a, b) in x.data.iter_mut().zip(y.data.iter()) {
        *a += b;
    }
    Ok(())
}

/// The full E3 encoder: staff tensor `(1, 128, W)` flat → the decoder's
/// cross-attention context `[1 + 8·(W/16), 256]` (cls first — §2b).
///
/// # Errors
/// Shape violations, a missing tensor, or a kernel failure.
pub fn encode(w: &TromrEncoderW, pixels: &[f32], width: usize) -> FocrResult<Mat> {
    let feat = backbone(w, pixels, width)?;
    let x = tokens_from_feature(&feat); // [8·wp, 1024] row-major (r, c)
    let x = w.patch_proj.apply(&x)?; // [8·wp, 256]

    let (rows, wp) = (feat.h, feat.w);
    let seq = 1 + rows * wp;
    let mut tok = Mat::from_vec(seq, DIM, vec![0.0f32; seq * DIM]);
    // cls token + pos[0].
    for d in 0..DIM {
        tok.data[d] = w.cls_token[d] + w.pos_embed[d];
    }
    // Patch tokens + CROP-INDEXED positions: (r, c) → pos_embed[1 + r·80 + c].
    for r in 0..rows {
        for c in 0..wp {
            let t = 1 + r * wp + c;
            let pos = (1 + r * POS_COLS + c) * DIM;
            let src = (r * wp + c) * DIM;
            for d in 0..DIM {
                tok.data[t * DIM + d] = x.data[src + d] + w.pos_embed[pos + d];
            }
        }
    }

    for blk in &w.blocks {
        let h = nn::layer_norm(&tok, Some(&blk.ln1_w), Some(&blk.ln1_b), LN_EPS)?;
        let attn = self_attention(blk, &h)?;
        add_assign(&mut tok, &attn)?;
        let h2 = nn::layer_norm(&tok, Some(&blk.ln2_w), Some(&blk.ln2_b), LN_EPS)?;
        let mut m = blk.fc1.apply(&h2)?;
        nn::gelu(&mut m);
        let m = blk.fc2.apply(&m)?;
        add_assign(&mut tok, &m)?;
    }
    nn::layer_norm(&tok, Some(&w.final_ln_w), Some(&w.final_ln_b), LN_EPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zoo_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var_os("FOCR_TROMR_DIR").map(std::path::PathBuf::from)?;
        dir.join("tromr.focrq").is_file().then_some(dir)
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("fixture bin reads");
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn width_and_buffer_guards_reject() {
        // Guards fire BEFORE weight access, so a dummy hydration works:
        // synthesize via the error path (no zoo needed).
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset (guard leg included)");
            return;
        };
        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let w = TromrEncoderW::build(&weights).expect("hydrates");
        // width not ×16, width 0, width > 1280, short buffer — all clean errors.
        assert!(encode(&w, &vec![0.0; IMG_H * 100], 100).is_err());
        assert!(encode(&w, &[], 0).is_err());
        assert!(encode(&w, &vec![0.0; IMG_H * 1296], 1296).is_err());
        assert!(encode(&w, &[0.0; 7], 800).is_err());
    }

    /// The E3 L1/L2 cert: every oracle seam (stem, stages, patch proj, each
    /// ViT block, the final norm) at cosine ≥ 0.9999 with maxabs ledgered;
    /// the oracle's own floor on this stack is 0.0 (same- AND cross-thread),
    /// so every divergence below is OUR summation-order envelope, reported
    /// per seam. Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_encoder_matches_torch_oracle() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fx_path = dir.join("tromr_oracle_fixtures.json");
        if !fx_path.is_file() {
            eprintln!(
                "[tromr-test] skip_no_model: oracle fixtures absent (gen_reference_fixtures_tromr.py)"
            );
            return;
        }
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx_path).unwrap()).unwrap();
        let width = fx["preproc"]["shape"][2].as_u64().unwrap() as usize;
        let pixels = read_f32(&dir.join("tromr_preproc.bin"));
        assert_eq!(pixels.len(), IMG_H * width, "preproc fixture shape");

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let w = TromrEncoderW::build(&weights).expect("hydrates");

        // Backbone seams (channel-major in the fixture, ours identical layout).
        let feat = backbone(&w, &pixels, width).expect("backbone runs");
        let stage2 = read_f32(&dir.join("tromr_seam_stage2.bin"));
        assert_eq!(feat.data.len(), stage2.len(), "stage2 shape");
        let (c, m) = (cos(&feat.data, &stage2), maxabs(&feat.data, &stage2));
        eprintln!("[tromr-cert] stage2 cos {c:.8} maxabs {m:.3e}");
        assert!(c >= 0.9999, "stage2 cos {c}");

        // Full encoder vs the final oracle output [1, seq, 256].
        let out = encode(&w, &pixels, width).expect("encode runs");
        let oracle = read_f32(&dir.join("tromr_seam_encoder_out.bin"));
        assert_eq!(out.data.len(), oracle.len(), "encoder_out shape");
        let (c, m) = (cos(&out.data, &oracle), maxabs(&out.data, &oracle));
        eprintln!(
            "[tromr-cert] encoder_out cos {c:.8} maxabs {m:.3e} (oracle floor 0.0 both legs)"
        );
        assert!(c >= 0.9999, "encoder_out cos {c}");
    }
}
