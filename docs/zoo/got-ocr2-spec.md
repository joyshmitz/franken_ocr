# GOT-OCR2.0 — architecture spec (bead bd-3jo6.2.1 / B1)

Implementation-ready census of **GOT-OCR2.0** (`stepfun-ai/GOT-OCR2_0`; "General OCR
Theory", arXiv [2409.01704](https://arxiv.org/abs/2409.01704); **Apache-2.0**) for a
from-scratch pure-Rust CPU port in franken_ocr (epic `bd-3jo6`, sub-epic B). Every
load-bearing number is cited to a source file; this doc is self-contained so the
implementer never needs to reverse-engineer the model. **Source of truth = the
released weights repo files** (`config.json`, `modeling_GOT.py`, `got_vision_b.py`,
`tokenization_qwen.py`, `qwen.tiktoken`, `render_tools.py`) + the GitHub repo
`Ucas-HaoranWei/GOT-OCR2.0` + the paper.

> **Headline.** Encoder = **SAM-ViT-B** image encoder + a 2-conv "Vary" compressor →
> **256 tokens × 1024-dim**. Connector = a single **`Linear(1024→1024)`**. Decoder =
> **Qwen1.5/Qwen2-architecture 0.5B, DENSE** (24 layers, hidden 1024, **full MHA — no
> GQA**, SwiGLU, RoPE θ=1e6, **full causal — no sliding window**, tied embeddings).
> ~580 M params total; `model.safetensors` 1.43 GB bf16, single shard.

---

## 1. Top-level graph (LLaVA/Vary splice)

```
image (1024×1024×3, RGB)
  → Vision encoder  (SAM-ViT-B backbone + neck + net_2/net_3 compressor)  → 256 × 1024
  → Connector       (mm_projector_vary = Linear(1024→1024))               → 256 × 1024
  → splice into decoder input embeddings at the 256 <imgpad> slots,
    bracketed by <img> … </img>
  → Decoder         (Qwen-0.5B dense, 24L, hidden 1024)                   → text tokens
```

Param split (arXiv): vision encoder ≈ 80 M, connector ≈ 1.05 M, decoder ≈ 500 M, **total
≈ 580 M**. CLIP is **not** used.

---

## 2. Vision encoder — `got_vision_b.py::build_GOT_vit_b` (≈ SAM-ViT-B)

| field | value |
|---|---|
| input | 1024×1024×3 RGB |
| patch_size | 16 (PatchEmbed `Conv2d(3,768,k16,s16)`) → 64×64 grid |
| embed_dim | 768 |
| depth | 12 |
| num_heads | 12 (head_dim 64) |
| mlp_ratio | 4.0 (MLP hidden 3072), GELU |
| qkv_bias | True |
| use_rel_pos | True (SAM decomposed relative position embeddings in attention) |
| global_attn_indexes | [2, 5, 8, 11] (full global attn; other layers windowed) |
| window_size | 14 |
| out_chans | 256 (neck) |
| pos embed | learned absolute `pos_embed (1,64,64,768)` init-zeros, PLUS rel-pos in attn |
| norm / act | LayerNorm (not RMS); neck `LayerNorm2d`; GELU |

**Neck** (after the 12 blocks; input permuted `(B,64,64,768)→(B,768,64,64)`):
`Conv2d(768,256,k1,bias=False) · LayerNorm2d(256) · Conv2d(256,256,k3,s1,p1,bias=False) · LayerNorm2d(256)` → `(B,256,64,64)`.

**Compressor** (the 16× downsample to 256 tokens; the part NOT in stock SAM):
`net_2 = Conv2d(256,512,k3,s2,p1,bias=False)` (64→32), `net_3 = Conv2d(512,1024,k3,s2,p1,bias=False)` (32→16) → `(B,1024,16,16)` → flatten+permute → **256 × 1024**.

**vs SAM-ViT-B:** backbone is byte-for-byte SAM ViT-B image encoder; the two strided
convs + token-sequence consumption are the only deltas. No SAM prompt-encoder/mask-decoder.

## 3. Connector — `mm_projector_vary = nn.Linear(1024, 1024)` (bias=True, no act, no norm).

## 4. Language decoder — literal `config.json` (`GOTQwenForCausalLM`, Qwen2 arch)

| field | value | | field | value |
|---|---|---|---|---|
| hidden_size | **1024** | | rope_theta | **1000000.0** |
| num_hidden_layers | **24** | | rms_norm_eps | **1e-6** |
| num_attention_heads | **16** | | hidden_act | **silu** (SwiGLU) |
| num_key_value_heads | **16** (NO GQA) | | use_sliding_window | **false** (full causal) |
| head_dim | **64** | | tie_word_embeddings | **true** |
| intermediate_size | **2816** | | torch_dtype | bfloat16 |
| vocab_size | **151860** | | bos/eos/pad | **151643** `<|endoftext|>` |
| max_position_embeddings | **32768** | | im_start/end/patch | **151857 / 151858 / 151859** |

Structurally **Qwen1.5-0.5B** (Qwen2 modeling code), embedding resized to 151860 for the
3 image tokens. **Attention biases (Qwen2):** `q/k/v_proj` bias=True, `o_proj` bias=False,
MLP+norms unbiased → **verify against weight keys (OQ-2)**.

## 5. Special tokens + prompt templates

Control (Qwen tiktoken): `<|endoftext|>`=151643 (bos=eos=pad), `<|im_start|>`=151644,
`<|im_end|>`=151645. Image (GOT-added): `<img>`=151857, `</img>`=151858, `<imgpad>`=151859,
`image_token_len`=256.

**Image splice:** `"<img>" + "<imgpad>"×256 + "</img>" + "\n" + <instruction>`; at forward
the 256 `<imgpad>` (151859) embedding rows are overwritten by the 256 projected vision
features. Multi-crop → `<imgpad>×(256·ll)` where `ll` = #tiles (+thumbnail).

**Conversation (`conv_mpt`, MPT style; literal plain-OCR prompt):**
```
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user
<img>{<imgpad>×256}</img>
OCR: <|im_end|><|im_start|>assistant
```
(No whitespace between `…detail.<|im_end|>` and `<|im_start|>user`.)

**Per-mode instruction `qs`:** plain `'OCR: '`; formatted/markdown `'OCR with format: '`;
fine-grained box `str(bbox)+' '+'OCR: '` (bbox normalized to a **0–1000 integer grid**);
fine-grained color `'['+color+'] '+'OCR: '` (color ∈ red/green/blue); region-reference
`'OCR upon the patch reference: '`. **The structured outputs (chart / sheet-music `**kern`
/ SMILES / geometry-tikz / tables / math) are NOT separate prompts — all driven by
`'OCR with format: '`; the model picks the formalism from the image.**

## 6. Preprocessing — `GOTImageEvalProcessor(image_size=1024)`

`.convert('RGB')` → bicubic `Resize((1024,1024))` (**squash, NO aspect preserve**) →
`ToTensor` [0,1] CHW → `Normalize(mean=(0.48145466,0.4578275,0.40821073),
std=(0.26862954,0.26130258,0.27577711))` (OpenAI/CLIP constants).

**Multi-crop `dynamic_preprocess`** (`run_ocr_2.0_crop.py`): `min_num=1,max_num=6,
image_size=1024,use_thumbnail=True`; candidate grids `(i,j)` with `1≤i·j≤6`; pick the
`(i,j)` whose `i/j` best matches `w/h`; resize to `(1024·i,1024·j)`, crop row-major into
`i·j` 1024² tiles; if `i·j>1` append one global thumbnail. **Max 6+1 tiles = 1792 image
tokens.**

## 7. Tokenizer — Qwen tiktoken BPE (NOT a HF `tokenizer.json`)

Original Qwen (`QWenTokenizer`, `tokenization_qwen.py`): merge-rank file **`qwen.tiktoken`**
(2.56 MB, `base64_token<space>rank` per line), the GPT-2/tiktoken regex pre-tokenizer, the
dual special-token sets. vocab 151860. **No `vocab.json`/`merges.txt`/`tokenizer.json`** →
the Rust port needs a **tiktoken-style byte-BPE loader** for `qwen.tiktoken` + the Qwen regex
+ specials (A6/B6; the standard `tokenizers`-crate JSON path will NOT load this).

## 8. Output postprocessing — `modeling_GOT.py` / `render_tools.py`

Generation (hard-coded in `chat`): `do_sample=False, num_beams=1, no_repeat_ngram_size=20,
max_new_tokens=4096`, stop string `"<|im_end|>"`. Greedy/deterministic. Plain `ocr`: strip
trailing `<|im_end|>` + a small punctuation-normalization table. `format`: raw output is
**Mathpix-Markdown (.mmd)** (Markdown + LaTeX math); rendering to HTML (MathJax / tikzjax /
SVG) is downstream. SMILES / `**kern` are emitted as plain text (render via RDKit/Verovio
externally — out of scope). Fine-grained box/color is an INPUT; output is plain/formatted
region text (no output coordinate decoding).

## 9. Open questions (doctrine hard rule — no kernel ships against an unresolved OQ)

- **OQ-1** param-count vs file size: 1.43 GB bf16 ≈ 715 M params vs 580 M headline — is
  `lm_head.weight` stored separately despite `tie_word_embeddings=true`? Inspect tensor keys.
- **OQ-2** Qwen2 attention biases: confirm `q/k/v_proj.bias` present, `o_proj.bias`/MLP bias absent.
- **OQ-3** vocab ids 151851–151856: exact tokens in the gap before `<img>`(151857).
- **OQ-4** SAM rel-pos exactness: decomposed rel-pos for BOTH 14×14 windowed and 64×64 global
  layers; `get_abs_pos`/window-pad logic must match.
- **OQ-5** vision LayerNorm + LayerNorm2d eps (likely 1e-6) — read from weights.
- **OQ-6** RoPE: full (not partial) rotary over head_dim 64.
- **OQ-7** exact bbox 0–1000 normalization (relative to original image vs the 1024 canvas).
- **OQ-8** `no_repeat_ngram_size=20` (HF builtin, GLOBAL not windowed) materially affects decode;
  interacts with the known int8 "repetition-runs on hard tables" failure mode — must implement to match.
- **OQ-9** tile batching: row-major grid order, thumbnail LAST, one flat `<img>…</img>` block.

## 10. Reuse map → franken_ocr beads

**Reuse near-as-is:**
- **SAM-ViT-B backbone** (`src/native_engine/vision_sam.rs`) → GOT vision backbone — same config
  (verify rel-pos, OQ-4). The CLIP tower (`vision_clip.rs`) is unused. → **B3**.
- **int8 GEMM kernels** (`ft-kernel-cpu` `linear_int8_dynamic_f32` + per-channel/row quant + NR4
  pack; NEON-SDOT / AVX-VNNI) → the Qwen2 decoder q/k/v/o + SwiGLU gate/up/down, and vision
  linears. Decoder-arch-agnostic. → **A7 / B5**. (Heed: int8 decode = 2.5× win but repetition-runs
  on hard tables — pair with OQ-8.)

**NEW (build):**
- Vision compressor convs `net_2`/`net_3` (256→512→1024, stride-2) → **B3 / A8**.
- Connector `Linear(1024→1024)` + the `<imgpad>`-slot splice → **B4 / B7**.
- **Qwen2-0.5B DENSE decoder** (vs our DeepSeek-V2 MoE): dense SwiGLU FFN (no experts/router),
  full MHA (16=16, head_dim 64), RMSNorm 1e-6, RoPE θ=1e6, tied embeddings, qkv biases, no SWA.
  int8 kernels drop in; the graph is new. → **A7 (shared dense decoder) + B5**.
- **Qwen tiktoken BPE** loader → **A6 / B6**.
- Prompt builder (`conv_mpt` + per-mode `qs` + splice) + mode renderers → **B7**.
- Multi-crop `dynamic_preprocess` (min1/max6 grid + 1024 tiling + thumbnail) → **B7 (preprocess)**.

## 11. Conversion / quant plan (B2)

`.focrq` for GOT-OCR2 under the doctrine-#2 policy: **int8** the Qwen2 decoder GEMMs
(q/k/v/o, gate/up/down); **high-precision (bf16/f32)** the SAM-ViT encoder, neck, compressor
convs, the connector, `embed_tokens`/`lm_head` (tied), and ALL norms. Tied embeddings → store
ONE matrix (pending OQ-1). The convert path applies this per-arch quant policy (A3) keyed by
the `got-ocr2` model id.

### Sources
- config.json / modeling_GOT.py / got_vision_b.py / tokenization_qwen.py / qwen.tiktoken /
  render_tools.py — https://huggingface.co/stepfun-ai/GOT-OCR2_0/tree/main
- GitHub (conversation.py, run_ocr_2.0*.py, README) — https://github.com/Ucas-HaoranWei/GOT-OCR2.0
- Paper — https://arxiv.org/abs/2409.01704
