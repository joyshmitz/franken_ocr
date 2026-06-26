# Vision-tower parity: franken_ocr vs baidu/Unlimited-OCR

The native vision tower (`vision_sam` → `vision_clip` → `vision_bridge`) was wired
to the real `Weights` accessors and validated **stage-by-stage against the pinned
baidu reference** (truth-pack modeling code, rev `3a7f4dbb`, bf16) on a real
scanned page (`royalnavy02clow.pdf` p.9, 200 DPI).

## Method
1. `scripts/baseline/dump_stage_activations.py` forward-hooks baidu's
   `sam_model`/`vision_model`/`projector` and dumps `sam_in/sam_out/clip_out/
   projector_out` as `.npy` (baidu bf16, cast f32).
2. `examples/{vision_dump,full_vision_dump}.rs` run franken_ocr's wired forwards
   on baidu's **exact** `sam_in` (decoupling preprocessing from vision math) and
   dump the same boundaries.
3. Cosine + max/mean abs-diff per stage.

## Results (page 9)
| Stage | Entry point | Cosine | mean\|Δ\| | mean/std (baidu vs franken) |
|---|---|---|---|---|
| Preprocess | `preprocess_image(Base{1024})` | 0.998915 | 0.0027 | channel means within 0.0014 |
| SAM ViT-B | `vision_sam::forward` | **0.999915** | 0.00078 | -0.0015/0.0892 vs -0.0015/0.0892 |
| CLIP-L/14 | `vision_clip::forward` | **0.999214** | 0.0047 | 0.0006/0.2019 vs 0.0006/0.2020 |
| Projector (full tower) | `vision_bridge::forward` | **0.999638** | 0.0024 | -0.0043/0.1461 vs -0.0043/0.1461 |

The full vision tower (the decoder-ready `[256,1280]` image embeddings) matches
baidu at **cosine 0.9996**. Residuals are bf16(baidu)→fp32(franken) numeric drift
(the engine is fp32 here; quantized kernels are a separate, later comparison).

## Remaining for end-to-end OCR
The decoder (DeepSeek-V2 MoE 12L) + lm_head + the prompt vision-token scatter +
the incremental decode KV-cache (see `docs/forward_wiring_intel.md`). The baidu
oracle for all 20 pages is generated under the baseline workspace; once the
decoder path is wired, `scripts/baseline/compare_ocr.py` scores CER end-to-end.
