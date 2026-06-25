# Phase -1 Truth Pack — Preprocess & Infer OQs (line-backed)

Source pinned at HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`.
Local snapshots: `/Users/jemanuel/projects/franken_ocr/docs/truth-pack/snapshots/`.

Files read in full:
- `modeling_unlimitedocr.py` (1299 lines)
- `conversation.py` (281 lines)
- `processor_config.json`
- (corroborating) `README.md`, `tokenizer_config.json`, `special_tokens_map.json`

Every claim below quotes the actual source line(s) with `file:line`.

---

## OQ-7 — Gundam `dynamic_preprocess` / `find_closest_aspect_ratio`: `min_num`/`max_num` defaults + per-aspect-ratio tile/token math

**Question (verbatim):** Gundam `dynamic_preprocess`/`find_closest_aspect_ratio` `min_num`/`max_num` defaults + per-aspect-ratio tile/token math — quote the function.

### ANSWER (definitive)

**Defaults:** `min_num=2, max_num=32, image_size=640, use_thumbnail=False`. The Gundam path (`infer(..., crop_mode=True)`) calls `dynamic_preprocess(image)` with **no overrides**, so these defaults are authoritative for Gundam.

`modeling_unlimitedocr.py:175`
```python
def dynamic_preprocess(image, min_num=2, max_num=32, image_size=640, use_thumbnail=False):
```

The Gundam call site passes only the image (defaults apply), and crop is only attempted when the image exceeds 640×640:

`modeling_unlimitedocr.py:859-868`
```python
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                else:
                    if crop_mode:
                        # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                        images_crop_raw, crop_ratio = dynamic_preprocess(image)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]
```

### Tile-grid (aspect-ratio candidate) generation

Candidate `(width_tiles, height_tiles)` grids are all `(i, j)` with `min_num <= i*j <= max_num`, sorted by tile-count `i*j`:

`modeling_unlimitedocr.py:180-184`
```python
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
```
With defaults `min_num=2, max_num=32` this yields **118 candidate grids**, blocks ranging **2 to 32** tiles (verified by executing the exact expression). `(1,1)` is NOT a candidate (it is excluded because `1*1 = 1 < min_num=2`); the no-crop case is handled separately by the `<= 640` short-circuit at line 859.

### Closest-aspect-ratio selection (tie-break by area)

`modeling_unlimitedocr.py:158-172`
```python
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio
```
Note: `aspect_ratio = orig_width / orig_height` (line 177), and the tie-break (line 169) prefers the larger grid only when original `area > 0.5 * image_size^2 * i * j`.

### Resize + tile slicing (per-tile geometry)

`modeling_unlimitedocr.py:192-213`
```python
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio
```
The image is resized to `(640*W, 640*H)` and sliced into a `W×H` grid of `640×640` tiles (row-major). `use_thumbnail` defaults `False`, so **no extra thumbnail tile** is appended in Gundam. Returns `(processed_images, target_aspect_ratio)` where `target_aspect_ratio == (W, H)`, captured as `crop_ratio` at the call site (line 865) and stored as `width_crop_num, height_crop_num = crop_ratio` (line 890).

### Per-aspect-ratio TOKEN math (Gundam, image_size=640)

Local-view image tokens are emitted only when `width_crop_num > 1 or height_crop_num > 1`:

`modeling_unlimitedocr.py:904, 915-917`
```python
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                ...
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                num_queries * height_crop_num)
```
With `patch_size=16, downsample_ratio=4, image_size=640` (lines 827-828, 901): `num_queries = ceil((640//16)/4) = ceil(40/4) = 10`. Local block tokens = `(10*W + 1) * (10*H)` (verified by execution):

| grid W×H | local tokens `(10W+1)(10H)` | + global(273) = total Gundam image tokens |
|---|---|---|
| 2×1 | 210 | 483 |
| 1×2 | 220 | 493 |
| 2×2 | 420 | 693 |
| 3×2 | 620 | 893 |
| 4×4 | 1640 | 1913 |

(Global view = 273; see OQ-18 for derivation.) The `+1` per local row is the `image_newline`/separator token; placeholder geometry mirrors the vision-feature assembly in `UnlimitedOCRModel.forward` (lines 534-538).

**UNBLOCKS:** the `dynamic_preprocess` / tiling kernel (Gundam preprocessing bead) and the image-placeholder-token-assembly kernel (defines exact `images_seq_mask` length and `<image>` placeholder count per crop grid).

---

## OQ-8 — Prompt-mode taxonomy (free-OCR vs layout vs grounding vs markdown) + whether `<|ref|>`/`<|det|>` bboxes are emitted in all modes

**Question (verbatim):** the prompt-mode taxonomy from conversation.py/the processor: free-OCR vs layout vs grounding vs markdown, and whether `<|ref|>`/`<|det|>` bboxes are emitted in all modes.

### ANSWER

**There is no enumerated "mode" switch in the code.** The "mode" is purely the free-text `prompt` string the caller passes to `infer()`/`infer_multi()`. The model is prompt-conditioned; the Python wrapper does **not** branch on prompt content to change behavior (the only prompt-content branch is `if 'line_type' in outputs` at line 1097, which is an *output*-content check for geometry plotting, not a prompt mode). The source contains a set of **example/commented prompt strings** that constitute the de-facto taxonomy:

`modeling_unlimitedocr.py:797-802` (commented alternatives inside `infer`)
```python
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
```

`modeling_unlimitedocr.py:278` (docstring example — markdown conversion)
```python
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
```

README example prompts (corroborating, same snapshot):
- `README.md:91` `prompt='<image>document parsing.'` (Gundam single-image)
- `README.md:103,129` / `modeling_unlimitedocr.py:1142,1147` `prompt='<image>Multi page parsing.'` (multi-image / PDF, base mode)

**De-facto prompt taxonomy (all are free-text, single `<image>` token + instruction):**
1. **Grounding / layout** — `"<image>\n<|grounding|>Given the layout of the image. "` (uses the `<|grounding|>` special token, id 128820).
2. **Free OCR** — `"<image>\nFree OCR. "`.
3. **Figure/element parse** — `"<image>\nParse the figure. "`.
4. **Plain text extraction** — `"<image>\nExtract the text in the image. "`.
5. **Markdown conversion** — `"<image_placeholder>\nExtract all information from this image and convert them into markdown format."`.
6. **Document parsing** — `"<image>document parsing."` (README, Gundam).
7. **Multi-page parsing** — `"<image>Multi page parsing."` (README / `infer_multi`).

### Special tokens that enable grounding bboxes

The grounding/ref/det tokens exist in the vocabulary:

`tokenizer_config.json:6558-6593`
```
"128816": { "content": "<|ref|>",
"128817": { "content": "<|/ref|>",
"128818": { "content": "<|det|>",
"128819": { "content": "<|/det|>",
"128820": { "content": "<|grounding|>",
```
(`<image>` itself is id 128815 — `tokenizer_config.json:6550-6551`, matching `image_token_id = 128815` at `modeling_unlimitedocr.py:845`/`1181`.) Note: `<|grounding|>`, `<|ref|>`, `<|det|>` are NOT in `special_tokens_map.json` (only User/Assistant/bos/eos/pad are), so they are added-vocab tokens, not "special map" entries.

### Are `<|ref|>`/`<|det|>` bboxes emitted in ALL modes?

**No — they are model-emitted output, present only when the model produces them (driven by the prompt, e.g. grounding/layout/document-parsing), and the wrapper only POST-PROCESSES them when `save_results=True`.** Bbox parsing/drawing is gated on output content, not forced:

`modeling_unlimitedocr.py:44-50` (`re_match` parses ref/det spans FROM the decoded output):
```python
def re_match(text):
    ref_pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(ref_pattern, text, re.DOTALL)
    det_pattern = r'(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)'
```

`modeling_unlimitedocr.py:1069-1082` (only on `save_results`, applied to whatever the model emitted):
```python
        if '<image>' in conversation[0]['content'] and save_results:
            outputs = tokenizer.decode(...)
            ...
            matches_ref, matches_images, mathes_other = re_match(outputs)
            result = process_image_with_refs(image_draw, matches_ref, output_path)
```
Coordinates are normalized to `[0,999]` (de-normalized at draw time, `modeling_unlimitedocr.py:107-111`: `x1 = int(x1 / 999 * image_width)`). If the model emits no `<|ref|>…<|det|>[…]<|/det|>` spans (e.g. plain "Free OCR." that returns prose/markdown), `re_match` returns empty and no boxes are drawn. So bbox emission is **mode/prompt-dependent, not universal**.

**UNBLOCKS:** the prompt-template / chat-formatting kernel (confirms prompts are free text wrapped by the `plain` template, no mode enum), and the grounding-bbox post-processing kernel (`re_match` regexes + 0–999 coordinate de-normalization), and the special-token-id table bead.

---

## OQ-17 — Does `infer()` force `.cuda()` / CUDA-autocast? (CPU-vs-GPU oracle)

**Question (verbatim):** does `infer()` force `.cuda()`/CUDA-autocast? quote the device + autocast lines — this decides the CPU-vs-GPU oracle.

### ANSWER (definitive): YES — both `infer()` and `infer_multi()` hard-code `.cuda()` on every input tensor and wrap generation in `torch.autocast("cuda", dtype=torch.bfloat16)`. There is NO device parameter and NO CPU fallback.

`infer()` non-eval path — `modeling_unlimitedocr.py:1002-1020`:
```python
            gen_kwargs = dict(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=[(images_crop.cuda(), images_ori.cuda())],
                images_seq_mask=images_seq_mask.unsqueeze(0).cuda(),
                images_spatial_crop=images_spatial_crop,
                ...
            )
            ...
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = self.generate(**gen_kwargs)
```

`infer()` eval_mode path — `modeling_unlimitedocr.py:1027-1044` (identical `.cuda()` + `torch.autocast("cuda", ...)`):
```python
            gen_kwargs = dict(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=[(images_crop.cuda(), images_ori.cuda())],
                images_seq_mask=images_seq_mask.unsqueeze(0).cuda(),
                ...
            )
            ...
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = self.generate(**gen_kwargs)
```

Output decode also calls `.cuda()` to compute the prompt length — `modeling_unlimitedocr.py:1049`:
```python
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
```

`infer_multi()` — `modeling_unlimitedocr.py:1238-1243`:
```python
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=input_ids.unsqueeze(0).cuda(),
                    images=[(dummy_crop.cuda(), images_ori.cuda())],
                    images_seq_mask=images_seq_mask.unsqueeze(0).cuda(),
```

Additionally, the vision-feature scatter in the model forward also hard-codes `.cuda()` on the mask — `modeling_unlimitedocr.py:582`:
```python
                    inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)
```

README confirms the model itself is moved to CUDA — `README.md:84`: `model = model.eval().cuda()`. Input dtype is `torch.bfloat16` (e.g. `modeling_unlimitedocr.py:886`).

**Implication for the oracle:** the reference (PyTorch) path is **GPU-only, bf16 autocast**. A CPU oracle cannot run this code unmodified — `.cuda()` calls will fail without a CUDA device, and even patched to CPU it runs under bf16 CUDA-autocast semantics. The franken_ocr CPU port must treat the reference as a GPU/bf16 oracle and account for bf16 vs fp32 numeric drift; an exact bit-match is not expected. The `images_spatial_crop` tensor is the one input deliberately left on CPU (no `.cuda()` at lines 1006/1031/1244).

**UNBLOCKS:** the oracle-harness design bead (must run the reference on a CUDA box, bf16; tolerance-based comparison, not bitwise) and the numeric-parity / dtype-policy decision for the CPU port.

---

## OQ-18 — Full per-mode token census + `no_repeat_ngram_size=35` + per-mode `ngram_window`

**Question (verbatim):** the FULL per-mode token census: the image-placeholder assembly = (16+1)*16+1=273 at base 1024? quote it; the Gundam multi-tile totals; and the `no_repeat_ngram_size=35` + ngram_window values per mode — 128 single-image vs 1024 multi-image — quote infer/infer_multi.

### ANSWER

#### (a) Base global-view placeholder = 273 at base_size=1024 — CONFIRMED

Constants: `patch_size = 16`, `downsample_ratio = 4` (`modeling_unlimitedocr.py:827-828`).

`num_queries_base` and the global-view placeholder assembly — `modeling_unlimitedocr.py:905, 913-914`:
```python
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
                ...
                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
```
At `base_size=1024`: `num_queries_base = ceil((1024//16)/4) = ceil(64/4) = 16`. Assembly = `([id]*16 + [id]) * 16 + [id]` = `(16+1)*16 + 1 = 272 + 1 = **273**`. **This is exactly `(16+1)*16+1=273`.** (verified by execution). The `+[id]` is the trailing view-separator token; the inner `+[id]` per row is the per-row `image_newline`.

This 273 global block is present in **both** the crop branch (lines 913-914) and the non-crop branch uses the same `(num_queries+1)*num_queries+1` shape with `num_queries` from `image_size` (line 950-953).

#### (b) Non-crop "base" mode (single image, image_size=1024) — `modeling_unlimitedocr.py:950-953`
```python
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                tokenized_image += [image_token_id]
```
At `image_size=1024`: `num_queries = 16` → **273 tokens** total (global only, no local tiles). This is the README "base" single-image config (`base_size=1024, image_size=1024, crop_mode=False`, `README.md:88`).

#### (c) Gundam (crop_mode=True) multi-tile totals — `modeling_unlimitedocr.py:913-917`
```python
                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                num_queries * height_crop_num)
```
Global = 273 (`num_queries_base=16`). Local (only if grid > 1×1), with `num_queries=10` (image_size=640): `(10*W + 1) * (10*H)`. Totals (= 273 + local), verified by execution:

| Gundam grid W×H | local `(10W+1)(10H)` | TOTAL image tokens (273 + local) |
|---|---|---|
| 1×1 (no crop, ≤640) | 0 | 273 |
| 2×1 | 210 | 483 |
| 1×2 | 220 | 493 |
| 2×2 | 420 | 693 |
| 3×2 | 620 | 893 |
| 4×4 | 1640 | 1913 |

Max grid 32 tiles → up to ~`273 + (varies by W,H)`; the per-grid value is `(10W+1)(10H)+273`.

#### (d) Multi-image (`infer_multi`, base, image_size=1024) per-image census — `modeling_unlimitedocr.py:1190, 1209-1212`
```python
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        ...
            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]  # separator token between images
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
```
At `image_size=1024`: `num_queries=16` → **273 tokens per image** (`(16+1)*16+1`), concatenated for N images at the single `<image>` position. No crop/local tiles in multi-image (it sets `images_spatial_crop.append([1, 1])` at line 1206; comment at line 1141 "Does NOT support crop mode").

#### (e) `valid_img_tokens` (compression-ratio accounting, NOT placeholder count)

Separate from the placeholder census, the code tracks `valid_img_tokens` for compression-ratio reporting — `modeling_unlimitedocr.py:875-878, 901-902`:
```python
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                ...
                if image_size == 640:
                    valid_img_tokens += len(images_crop_list) * 100
```
where `ratio = 1 - ((max(w,h) - min(w,h)) / max(w,h))` (line 838). Base 1024 contributes `256*ratio`; each 640 crop tile contributes 100. (This is a heuristic compression metric, distinct from the literal placeholder token count above.)

#### (f) `no_repeat_ngram_size=35` + per-mode `ngram_window`

`no_repeat_ngram_size` and `ngram_window` are **function params**, both default 0 (disabled) in code:

`infer` signature — `modeling_unlimitedocr.py:787`:
```python
    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False, max_length=32768, tps_interval=0, no_repeat_ngram_size=0, ngram_window=0, temperature=0.0):
```

`infer_multi` signature — `modeling_unlimitedocr.py:1139`:
```python
    def infer_multi(self, tokenizer, prompt='', image_files=None, output_path='', image_size=640, save_results=False, max_length=32768, tps_interval=0, no_repeat_ngram_size=0, ngram_window=0, temperature=0.0):
```

The **recommended per-mode values come from the README usage**, not hard-coded defaults:
- **Single-image (Gundam/base):** `no_repeat_ngram_size=35, ngram_window=128` — `README.md:96` and SGLang `README.md:250` (`ngram_window=128`).
- **Multi-image / PDF (base):** `no_repeat_ngram_size=35, ngram_window=1024` — `README.md:108` and `README.md:134`; SGLang `README.md:253,256` (`ngram_window=1024`).

How the window/size are applied (sliding-window n-gram blocker) — `modeling_unlimitedocr.py:1014-1017` (mirrored at 1038-1041 eval, and 1252-1255 in `infer_multi`):
```python
            if no_repeat_ngram_size > 0 and ngram_window > 0:
                gen_kwargs['logits_processor'] = [SlidingWindowNoRepeatNgramProcessor(no_repeat_ngram_size, ngram_window)]
            elif no_repeat_ngram_size > 0:
                gen_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size
```

The processor itself (window = lookback length; ngram_size = n) — `modeling_unlimitedocr.py:354-383`:
```python
class SlidingWindowNoRepeatNgramProcessor:
    """Block n-gram repetitions within a sliding window.
    Aligned with SGLang DeepseekOCRNoRepeatNGramLogitProcessor."""
    def __init__(self, ngram_size, window, whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window = window
        ...
    def __call__(self, input_ids, scores):
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < self.ngram_size:
                continue
            search_start = max(0, len(sequence) - self.window)
            search_end = len(sequence) - self.ngram_size + 1
            ...
            for token_id in banned:
                scores[batch_idx, token_id] = float('-inf')
        return scores
```
SGLang side uses the same numbers — `README.md:219-220`: `"ngram_size": 35, "window_size": ngram_window`.

`max_length=32768` is the generation cap in all paths (`modeling_unlimitedocr.py:787, 1011, 1139, 1249`), matching `README.md` and `--context-length 32768`.

**Summary table (mode → params):**

| Mode | function | base_size | image_size | crop_mode | ngram_size | ngram_window |
|---|---|---|---|---|---|---|
| Gundam single | `infer` | 1024 | 640 | True | 35 | 128 |
| Base single | `infer` | 1024 | 1024 | False | 35 | 128 |
| Multi-image / PDF | `infer_multi` | — | 1024 | n/a (no crop) | 35 | 1024 |

**UNBLOCKS:** the generation/sampling kernel (`SlidingWindowNoRepeatNgramProcessor` reimplementation with exact `search_start/search_end` semantics), the per-mode generation-config bead (ngram_size=35, window 128 vs 1024, max_length=32768, temperature=0 greedy → `do_sample=False`), and the image-placeholder-token-assembly kernel (273 base block, crop/local geometry, multi-image concatenation).

---

## Notes / what was needed but not in scope here

- `tokenizer.json` (raw BPE merges/vocab) is NOT present locally; token *IDs* used by the code (`image_token_id=128815`, bos_id=0, ref/det/grounding 128816–128820, eos `<｜end▁of▁sentence｜>`) are confirmed from `tokenizer_config.json` and inline constants. Exact byte-level tokenization of arbitrary text strings cannot be reproduced from these snapshots alone (would need `tokenizer.json` / the BPE model) — relevant if a downstream bead needs to reproduce text-split token counts exactly. Not a blocker for OQ-7/8/17/18 (all answered from present source).
