//! HUD / legend text rasterization.
//!
//! Renders short ASCII / Unicode lines (legend overlay, memo HUD) to an
//! RGBA8 buffer the renderer uploads as a 2D texture. Backed by an embedded
//! Inter Regular TTF rasterized at runtime via `ab_glyph`. Replaces the
//! prior 5×7 hand-rolled bitmap font (hash-thing-bujj).
//!
//! Output convention:
//! - Glyph color: warm off-white `(245, 233, 214)`, **unmodulated** by
//!   coverage. The alpha channel carries the rasterizer's coverage value
//!   (0..=255). Background pixels stay fully transparent.
//! - This produces correct anti-aliasing against the legend panel under
//!   `Rgba8UnormSrgb` blending: alpha is gamma-blended by the hardware,
//!   color stays at its sRGB-encoded target.
//!
//! Sizing is driven by the caller's `scale` argument. The font is
//! rasterized directly at the final size — there is no nearest-neighbor
//! upscale pass.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use std::sync::LazyLock;

/// Embedded Inter Regular (SIL Open Font License 1.1).
/// See `assets/fonts/Inter-LICENSE.txt`.
const INTER_REGULAR: &[u8] = include_bytes!("../assets/fonts/Inter-Regular.ttf");

static FONT: LazyLock<FontRef<'static>> = LazyLock::new(|| {
    FontRef::try_from_slice(INTER_REGULAR)
        .expect("Inter-Regular.ttf is a valid TrueType font (build asset, fail-loud)")
});

/// Native cap-height target before `scale`. The 5×7 bitmap that this
/// replaced rendered at ~9 px cap-height post-scale=2; matching that
/// keeps the legend panel close to its prior on-screen footprint.
const NATIVE_CAP_HEIGHT_PX: f32 = 9.0;

/// Native padding (per side) before `scale`.
const NATIVE_PADDING_PX: u32 = 2;

/// Glyph color (sRGB-encoded, unmodulated).
const GLYPH_R: u8 = 245;
const GLYPH_G: u8 = 233;
const GLYPH_B: u8 = 214;

/// Render lines of text into an RGBA pixel buffer.
///
/// Returns `(pixels, width, height)`. Background is transparent so the
/// shader can supply the panel treatment; glyphs are warm off-white.
/// `scale` multiplies the rasterization size — pass `2` for the legend
/// panel.
pub fn render_text_rgba(lines: &[&str], scale: u32) -> (Vec<u8>, u32, u32) {
    let scale = scale.max(1);
    let pad = NATIVE_PADDING_PX * scale;

    let px_scale = PxScale::from(NATIVE_CAP_HEIGHT_PX * scale as f32);
    let scaled_font = FONT.as_scaled(px_scale);

    let ascent = scaled_font.ascent();
    let descent = scaled_font.descent();
    let line_gap = scaled_font.line_gap();
    let line_height = ascent - descent;
    let line_pitch = line_height + line_gap;

    // Width: longest laid-out line. Layout uses char-driven advance so
    // multi-byte UTF-8 cannot inflate the texture (hash-thing-9q5o).
    let max_advance = lines
        .iter()
        .map(|line| line_advance(line, &scaled_font))
        .fold(0.0f32, f32::max);

    // Height: lines * pitch − one trailing line_gap, plus ascent above
    // the first baseline and |descent| below the last.
    let n_lines = lines.len() as f32;
    let content_h = if n_lines > 0.0 {
        n_lines * line_height + (n_lines - 1.0).max(0.0) * line_gap
    } else {
        0.0
    };

    let tex_w = (max_advance.ceil() as u32) + 2 * pad;
    let tex_h = (content_h.ceil() as u32) + 2 * pad;

    let out_w = tex_w.max(2 * pad).max(1);
    let out_h = tex_h.max(2 * pad).max(1);

    let mut pixels = vec![0u8; (out_w * out_h * 4) as usize];

    if lines.is_empty() || lines.iter().all(|l| l.is_empty()) {
        return (pixels, out_w, out_h);
    }

    for (row, line) in lines.iter().enumerate() {
        let baseline_y = pad as f32 + ascent + row as f32 * line_pitch;
        let mut pen_x = pad as f32;

        for ch in line.chars() {
            let glyph_id = scaled_font.glyph_id(ch);
            let glyph =
                glyph_id.with_scale_and_position(px_scale, ab_glyph::point(pen_x, baseline_y));
            let advance = scaled_font.h_advance(glyph_id);

            if let Some(outlined) = scaled_font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                let bx = bounds.min.x;
                let by = bounds.min.y;
                outlined.draw(|gx, gy, coverage| {
                    let px = (bx + gx as f32).round() as i32;
                    let py = (by + gy as f32).round() as i32;
                    if px < 0 || py < 0 {
                        return;
                    }
                    let px = px as u32;
                    let py = py as u32;
                    if px >= out_w || py >= out_h {
                        return;
                    }
                    let idx = ((py * out_w + px) * 4) as usize;
                    let new_a = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
                    let cur_a = pixels[idx + 3];
                    if new_a > cur_a {
                        pixels[idx] = GLYPH_R;
                        pixels[idx + 1] = GLYPH_G;
                        pixels[idx + 2] = GLYPH_B;
                        pixels[idx + 3] = new_a;
                    }
                });
            }

            pen_x += advance;
        }
    }

    (pixels, out_w, out_h)
}

fn line_advance<F: Font>(line: &str, scaled_font: &ab_glyph::PxScaleFont<&F>) -> f32 {
    line.chars()
        .map(|c| scaled_font.h_advance(scaled_font.glyph_id(c)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Output texture dimensions match the pixel buffer length.
    #[test]
    fn render_produces_correct_dimensions() {
        let (pixels, w, h) = render_text_rgba(&["Hello", "World!"], 2);
        assert!(w > 0, "width must be positive");
        assert!(h > 0, "height must be positive");
        assert_eq!(pixels.len(), (w * h * 4) as usize);
    }

    #[test]
    fn empty_lines_produce_minimal_texture() {
        let (pixels, w, h) = render_text_rgba(&[], 1);
        // Just padding on both axes.
        assert!(w >= 2 * NATIVE_PADDING_PX);
        assert!(h >= 2 * NATIVE_PADDING_PX);
        assert_eq!(pixels.len(), (w * h * 4) as usize);
        // No lit pixels in an empty render.
        let lit = (0..(pixels.len() / 4))
            .filter(|&i| pixels[i * 4 + 3] > 0)
            .count();
        assert_eq!(lit, 0);
    }

    #[test]
    fn letter_a_has_lit_pixels() {
        let (pixels, w, _h) = render_text_rgba(&["A"], 1);
        let lit_count = (0..(pixels.len() / 4))
            .filter(|&i| pixels[i * 4 + 3] > 0)
            .count();
        assert!(
            lit_count > 0,
            "Expected some lit glyph pixels for 'A', tex_w={w}"
        );
    }

    /// Texture sizing is driven by char count, not UTF-8 byte length
    /// (hash-thing-9q5o). `"Aéé"` is 3 chars / 5 bytes; `"AAA"` is
    /// 3 chars / 3 bytes. Both must render at comparable widths — far
    /// closer to each other than a `byte_len * per_char_advance` sizing
    /// would produce (which would size `"Aéé"` ≈ 5/3 wider than `"AAA"`).
    #[test]
    fn non_ascii_sizes_by_char_count_not_byte_len() {
        let (_, w_aaa, _) = render_text_rgba(&["AAA"], 1);
        let (_, w_aee, _) = render_text_rgba(&["Aéé"], 1);
        // Both renders succeeded with positive widths.
        assert!(w_aaa > 0 && w_aee > 0);
        // Character widths vary in proportional fonts, but byte-length
        // sizing would inflate "Aéé" by ~5/3 = 1.67×. Both widths must
        // stay within a 1.5× factor of each other.
        let ratio = w_aaa.max(w_aee) as f32 / w_aaa.min(w_aee) as f32;
        assert!(
            ratio < 1.5,
            "char-driven layout should keep widths within 1.5×; \
             got w_aaa={w_aaa}, w_aee={w_aee}, ratio={ratio:.3}"
        );
    }

    /// Glyph fill color is constant; only the alpha channel carries
    /// coverage. Sample any lit pixel from a rendered 'A' and verify
    /// its RGB matches the documented constants.
    #[test]
    fn lit_pixels_have_constant_glyph_color() {
        let (pixels, _, _) = render_text_rgba(&["A"], 2);
        let mut found = false;
        for i in 0..(pixels.len() / 4) {
            if pixels[i * 4 + 3] > 0 {
                assert_eq!(pixels[i * 4], GLYPH_R);
                assert_eq!(pixels[i * 4 + 1], GLYPH_G);
                assert_eq!(pixels[i * 4 + 2], GLYPH_B);
                found = true;
                break;
            }
        }
        assert!(found, "expected at least one lit pixel");
    }
}
