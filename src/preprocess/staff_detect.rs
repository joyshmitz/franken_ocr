//! E5 (bd-3jo6.5.5): the TrOMR staff-detection front end — full page →
//! ordered single-staff crops (tromr-spec §7, v1 scope: printed/scanned
//! pages, GLOBAL deskew only; camera dewarp and barline-split chunking are
//! filed follow-ups).
//!
//! Classical CV, pure Rust, no new dependencies:
//!
//! 1. ink gray plane (the DISC-004 rule: inverted alpha only when alpha
//!    varies; else cv2 fixed-point luma) → Otsu binarization;
//! 2. global deskew by shear: the angle in ±5° that MAXIMIZES the row-
//!    projection variance (staff lines align → sharp peaks), coarse 1° then
//!    fine 0.25°;
//! 3. row-projection profile → staff-LINE bands (rows whose ink count clears
//!    half the profile peak), merged and centered;
//! 4. groups of 5 consecutive bands with near-uniform spacing (≤ 25%
//!    deviation) become STAVES;
//! 5. each staff is cropped full-width with a vertical margin of twice the
//!    line spacing × 2 (ledger lines, dynamics), clamped to the page.
//!
//! The detector returns crops top-to-bottom with their page bboxes — the
//! `staves[]` contract E9's full-page path and the multi-staff JSON ride.

use image::DynamicImage;

use crate::error::{FocrError, FocrResult};

/// One detected staff: the cropped ink-gray plane + its page-space bbox
/// (post-deskew coordinates).
pub struct StaffCrop {
    /// Row-major gray pixels (ink dark), `h × w`.
    pub gray: Vec<u8>,
    /// Crop width (the full deskewed page width).
    pub w: usize,
    /// Crop height.
    pub h: usize,
    /// `(x, y, w, h)` on the deskewed page.
    pub bbox: (usize, usize, usize, usize),
}

/// The ink gray plane (shared with `tromr_staff_tensor` — DISC-004 rule).
fn ink_gray(img: &DynamicImage) -> (Vec<u8>, usize, usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let alpha_is_ink = img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p.0[3] < 255);
    let gray = if alpha_is_ink {
        img.to_rgba8().pixels().map(|p| 255 - p.0[3]).collect()
    } else {
        img.to_rgb8()
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                ((4899 * u32::from(r) + 9617 * u32::from(g) + 1868 * u32::from(b) + 8192) >> 14)
                    .min(255) as u8
            })
            .collect()
    };
    (gray, w, h)
}

/// Otsu's threshold over a 256-bin histogram. The returned `t` is the LAST
/// value of the dark class (ink = `v <= t` — dark pixels on a light page).
fn otsu_threshold(gray: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in gray {
        hist[v as usize] += 1;
    }
    let total: u64 = gray.len() as u64;
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w_b, mut sum_b, mut best_t, mut best_var) = (0u64, 0.0f64, 0u8, -1.0f64);
    for (t, &count) in hist.iter().enumerate() {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as f64 * count as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum_all - sum_b) / w_f as f64;
        let var = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// Row-projection ink counts under a shear of `tan_a` (column x shifts down
/// by `tan_a · x` rows) — evaluated WITHOUT materializing the sheared image.
fn sheared_row_profile(ink: &[bool], w: usize, h: usize, tan_a: f64) -> Vec<u32> {
    let mut profile = vec![0u32; h];
    for y in 0..h {
        let row = &ink[y * w..(y + 1) * w];
        for (x, &is_ink) in row.iter().enumerate() {
            if is_ink {
                let shift = (tan_a * x as f64).round() as isize;
                let ny = y as isize - shift;
                if ny >= 0 && (ny as usize) < h {
                    profile[ny as usize] += 1;
                }
            }
        }
    }
    profile
}

fn profile_variance(profile: &[u32]) -> f64 {
    let n = profile.len() as f64;
    let mean = profile.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    profile
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n
}

/// The global deskew angle (degrees) in ±5° maximizing row-profile variance:
/// coarse 1° sweep then fine 0.25° around the winner.
fn deskew_angle(ink: &[bool], w: usize, h: usize) -> f64 {
    let score = |deg: f64| -> f64 {
        profile_variance(&sheared_row_profile(ink, w, h, deg.to_radians().tan()))
    };
    let mut best = (0.0f64, score(0.0));
    for d in -5..=5 {
        let deg = f64::from(d);
        let s = score(deg);
        if s > best.1 {
            best = (deg, s);
        }
    }
    let coarse = best.0;
    let mut fine = best;
    for i in -3..=3 {
        let deg = coarse + f64::from(i) * 0.25;
        let s = score(deg);
        if s > fine.1 {
            fine = (deg, s);
        }
    }
    fine.0
}

/// Shear the gray plane vertically by `-tan(angle)·x` (fills with 255 =
/// paper). Adequate for the ≤5° global-deskew scope.
fn shear_gray(gray: &[u8], w: usize, h: usize, deg: f64) -> Vec<u8> {
    if deg == 0.0 {
        return gray.to_vec();
    }
    let tan_a = deg.to_radians().tan();
    let mut out = vec![255u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let shift = (tan_a * x as f64).round() as isize;
            let ny = y as isize - shift;
            if ny >= 0 && (ny as usize) < h {
                out[ny as usize * w + x] = gray[y * w + x];
            }
        }
    }
    out
}

/// Merge threshold-passing rows into bands, returning each band's center row.
fn line_band_centers(profile: &[u32], min_count: u32) -> Vec<usize> {
    let mut centers = Vec::new();
    let mut start: Option<usize> = None;
    for (y, &c) in profile.iter().enumerate() {
        if c >= min_count {
            start.get_or_insert(y);
        } else if let Some(s) = start.take() {
            centers.push((s + y - 1) / 2);
        }
    }
    if let Some(s) = start {
        centers.push((s + profile.len() - 1) / 2);
    }
    centers
}

/// Group line centers into 5-line staves with near-uniform spacing (≤ 25%
/// max deviation from the mean gap). Greedy left-to-right — staves do not
/// overlap on a page.
fn group_staves(centers: &[usize]) -> Vec<[usize; 5]> {
    let mut staves = Vec::new();
    let mut i = 0;
    while i + 4 < centers.len() {
        let five = [
            centers[i],
            centers[i + 1],
            centers[i + 2],
            centers[i + 3],
            centers[i + 4],
        ];
        let gaps: Vec<f64> = five.windows(2).map(|p| (p[1] - p[0]) as f64).collect();
        let mean = gaps.iter().sum::<f64>() / 4.0;
        let ok = mean >= 2.0 && gaps.iter().all(|g| (g - mean).abs() <= 0.25 * mean);
        if ok {
            staves.push(five);
            i += 5;
        } else {
            i += 1;
        }
    }
    staves
}

/// Detect staves on a full page (tromr-spec §7 v1). Returns crops
/// top-to-bottom; an empty result means "no 5-line staff found" (the caller
/// decides whether to fall back to whole-image recognition).
///
/// # Errors
/// A degenerate (zero-sized) image.
pub fn detect_staves(img: &DynamicImage) -> FocrResult<Vec<StaffCrop>> {
    let (gray, w, h) = ink_gray(img);
    if w == 0 || h == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "staff_detect: degenerate {w}x{h} input"
        )));
    }
    let thr = otsu_threshold(&gray);
    let ink: Vec<bool> = gray.iter().map(|&v| v <= thr).collect();

    let angle = deskew_angle(&ink, w, h);
    let gray = shear_gray(&gray, w, h, angle);
    let ink: Vec<bool> = gray.iter().map(|&v| v <= thr).collect();

    let profile = sheared_row_profile(&ink, w, h, 0.0);
    let peak = profile.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        return Ok(Vec::new());
    }
    let centers = line_band_centers(&profile, peak / 2);
    let staves = group_staves(&centers);

    // The model's positional budget: a crop resized to h=128 may span at
    // most 1280 columns (tromr POS_COLS * PATCH). Band geometry below is
    // shaped so real full-width systems FIT instead of hard-failing the
    // clamp (bd-av64.14; measured 2026-07-06: full-page-width bands with
    // page margins included blew the budget on every dense real scan, while
    // recognition quality is insensitive to GENEROUS bands and catastrophic
    // only for over-TIGHT ones).
    let budget = crate::native_engine::tromr::POS_COLS * crate::native_engine::tromr::PATCH;
    let img_h = crate::native_engine::tromr::IMG_H;

    let mut crops = Vec::with_capacity(staves.len());
    for (i, five) in staves.iter().enumerate() {
        let spacing = (five[4] - five[0]) as f64 / 4.0;
        let margin = (2.0 * spacing).round() as usize * 2;
        // Neighbor-aware vertical bounds: a band may extend to the midline
        // toward the adjacent staff (page edge otherwise), so bands never
        // swallow a neighbor's lines no matter how far they extend.
        let lo_bound = if i > 0 {
            (staves[i - 1][4] + five[0]) / 2
        } else {
            0
        };
        let hi_bound = if i + 1 < staves.len() {
            (five[4] + staves[i + 1][0]).div_ceil(2)
        } else {
            h
        };
        let mut y0 = five[0].saturating_sub(margin).max(lo_bound);
        let mut y1 = (five[4] + margin + 1).min(h).min(hi_bound);

        // (1) Horizontal ink-extent trim: the staff lines span the system,
        // so any column inside it holds >= 5 ink pixels; a >= 2 floor keeps
        // isolated specks from stretching the band to the page margins.
        // Pad the ink extent by ~2 line-spacings each side (never tighter
        // than the ink itself).
        let col_ink = |x: usize, a: usize, b: usize| -> usize {
            (a..b).filter(|&y| gray[y * w + x] <= thr).count()
        };
        let mut x0 = 0;
        while x0 < w && col_ink(x0, y0, y1) < 2 {
            x0 += 1;
        }
        let mut x1 = w;
        while x1 > x0 && col_ink(x1 - 1, y0, y1) < 2 {
            x1 -= 1;
        }
        if x1 <= x0 {
            // No ink columns at all (cannot happen for a grouped staff, but
            // stay defensive): keep the full width.
            x0 = 0;
            x1 = w;
        }
        let pad = (2.0 * spacing).round() as usize;
        x0 = x0.saturating_sub(pad);
        x1 = (x1 + pad).min(w);

        // (2) Extend-to-fit: if the trimmed band still resizes past the
        // positional budget, grow it vertically toward the neighbor bounds
        // (measured: a staff occupying as little as ~30% of the frame still
        // reads correctly, while width overflow is a hard failure). A band
        // that cannot reach the budget within its bounds is emitted as-is —
        // the per-staff recovery (bd-av64.2) skips it with a named reason.
        let step = spacing.max(1.0).round() as usize;
        while img_h * (x1 - x0) > budget * (y1 - y0) && (y0 > lo_bound || y1 < hi_bound) {
            y0 = y0.saturating_sub(step).max(lo_bound);
            y1 = (y1 + step).min(hi_bound).min(h);
        }

        let (ch, cw) = (y1 - y0, x1 - x0);
        let mut crop = vec![0u8; ch * cw];
        for (row, y) in (y0..y1).enumerate() {
            crop[row * cw..(row + 1) * cw].copy_from_slice(&gray[y * w + x0..y * w + x1]);
        }
        crops.push(StaffCrop {
            gray: crop,
            w: cw,
            h: ch,
            bbox: (x0, y0, cw, ch),
        });
    }
    Ok(crops)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draw a synthetic page: `staves` five-line groups (line thickness 2,
    /// spacing 10) at the given top offsets, on a 255 page with some noise
    /// notes (short dark runs) between lines.
    fn synth_page(w: usize, h: usize, staff_tops: &[usize]) -> DynamicImage {
        let mut gray = vec![250u8; w * h];
        for &top in staff_tops {
            for line in 0..5 {
                let y = top + line * 10;
                for dy in 0..2 {
                    for x in 20..w - 20 {
                        gray[(y + dy) * w + x] = 10;
                    }
                }
            }
            // a few "note heads" between the lines
            for k in 0..6 {
                let cx = 60 + k * 90;
                let cy = top + 14 + (k % 3) * 10;
                for dy in 0..5 {
                    for dx in 0..7 {
                        gray[(cy + dy) * w + cx + dx] = 30;
                    }
                }
            }
        }
        let img = image::GrayImage::from_raw(w as u32, h as u32, gray).unwrap();
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn detects_two_staves_in_order_with_sane_crops() {
        let img = synth_page(800, 400, &[80, 250]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 2, "two 5-line groups");
        // Top-to-bottom order + bboxes cover each staff with margin.
        assert!(crops[0].bbox.1 < crops[1].bbox.1);
        let (_, y0, _, ch0) = crops[0].bbox;
        assert!(y0 < 80 && y0 + ch0 > 80 + 40, "margin spans the staff");
        // Every crop is full-width and non-degenerate.
        for c in &crops {
            assert_eq!(c.w, 800);
            assert!(c.h >= 40 && c.h <= 200, "crop height {}", c.h);
            assert_eq!(c.gray.len(), c.w * c.h);
        }
    }

    #[test]
    fn deskew_recovers_a_sheared_page() {
        // Shear the synthetic page by ~2° and confirm detection still finds
        // both staves (the deskew must undo the tilt).
        let img = synth_page(800, 400, &[80, 250]);
        let gray = img.to_luma8();
        let sheared = shear_gray(gray.as_raw(), 800, 400, -2.0);
        let tilted =
            DynamicImage::ImageLuma8(image::GrayImage::from_raw(800, 400, sheared).unwrap());
        let crops = detect_staves(&tilted).expect("detects");
        assert_eq!(crops.len(), 2, "deskew recovers both staves");
    }

    #[test]
    fn blank_and_noise_pages_yield_no_staves() {
        let blank =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(400, 300, image::Luma([255u8])));
        assert!(detect_staves(&blank).expect("runs").is_empty());
        // 4 lines (not 5) must NOT group into a staff.
        let mut gray = vec![250u8; 400 * 300];
        for line in 0..4 {
            let y = 100 + line * 10;
            for x in 20..380 {
                gray[y * 400 + x] = 10;
            }
        }
        let four = DynamicImage::ImageLuma8(image::GrayImage::from_raw(400, 300, gray).unwrap());
        assert!(
            detect_staves(&four).expect("runs").is_empty(),
            "4 lines != a staff"
        );
    }

    /// bd-av64.14: horizontal ink-extent trim — page margins (columns with
    /// no ink) leave the band; the ink span plus ~2-spacing pads stays.
    #[test]
    fn trim_cuts_page_margins_but_keeps_ink() {
        // Hand-drawn staff with WIDE page margins: lines span x 200..600 on
        // an 800-wide page (spacing 10 => trim pad 20).
        let mut gray = vec![250u8; 800 * 260];
        for line in 0..5 {
            let y = 80 + line * 10;
            for dy in 0..2 {
                for x in 200..600 {
                    gray[(y + dy) * 800 + x] = 10;
                }
            }
        }
        let img =
            DynamicImage::ImageLuma8(image::GrayImage::from_raw(800, 260, gray).expect("synth"));
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 1);
        let (x0, _y0, cw, _ch) = crops[0].bbox;
        assert!(
            (150..=200).contains(&x0),
            "left margin trimmed to pad (x0 {x0})"
        );
        assert!(
            x0 + cw <= 650 && x0 + cw >= 600,
            "right margin trimmed (x0+cw {})",
            x0 + cw
        );
    }

    /// bd-av64.14: extend-to-fit — a wide staff with vertical room grows its
    /// band until the resized width fits the 1280 positional budget.
    #[test]
    fn wide_staff_with_room_fits_the_positional_budget() {
        let img = synth_page(2000, 700, &[320]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 1);
        let (_x0, _y0, cw, ch) = crops[0].bbox;
        assert!(
            128 * cw <= 1280 * ch,
            "resized width {} exceeds 1280 (cw {cw}, ch {ch})",
            128 * cw / ch
        );
    }

    /// bd-av64.14: neighbor bounds — two packed wide staves may extend only
    /// to their shared midline; bands never overlap even under budget
    /// pressure, and every band keeps the whole 5-line span.
    #[test]
    fn packed_staves_stop_at_the_midline() {
        let img = synth_page(2000, 400, &[100, 240]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 2);
        let (_, ay, _, ah) = crops[0].bbox;
        let (_, by, _, _bh) = crops[1].bbox;
        assert!(ay + ah <= by, "bands must not overlap ({ay}+{ah} vs {by})");
        assert!(ay <= 100 && ay + ah >= 140 + 2, "staff A span kept");
    }

    /// bd-av64.14: monotonic safety — with no budget pressure and no close
    /// neighbor, the band is never TIGHTER than the classic 12-spacing form.
    #[test]
    fn unpressured_band_keeps_the_generous_margins() {
        let img = synth_page(800, 400, &[180]);
        let crops = detect_staves(&img).expect("detects");
        let (_, _, _, ch) = crops[0].bbox;
        // spacing 10 => staff span 40 + 2 x 40 margins = 120.
        assert!(ch >= 120, "band height {ch} tighter than the classic form");
    }

    #[test]
    fn otsu_separates_bimodal() {
        let mut v = vec![20u8; 500];
        v.extend(vec![230u8; 500]);
        let t = otsu_threshold(&v);
        // Convention: ink = v <= t, so t must include the dark mode and
        // exclude the light one.
        assert!((20..230).contains(&t), "threshold {t} between the modes");
    }
}
