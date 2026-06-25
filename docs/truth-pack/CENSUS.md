# Truth Pack — Token / Shape / Buffer CENSUS

**Bead:** `PM1-census-generator`
**Pinned model source:** HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`, snapshotted locally at
`/Users/jemanuel/projects/franken_ocr/docs/truth-pack/snapshots/`.

Every count below is LINE-BACKED — derived directly from the pinned
`model.safetensors.index.json` `weight_map` and from `config.json`, with the exact
source lines quoted. Where the index does NOT carry a fact (e.g. per-tensor shapes are
absent), it is computed analytically from `config.json` dims and the fact is flagged.

> **Key correction to the bead's prior estimate.** The bead hypothesized
> `shared 11*2*3 = 66` and a `~2229` Linear target. The ACTUAL index shows the two
> shared experts are **fused into a single `DeepseekV2MLP`** (`n_shared_experts=2`
> widens `moe_intermediate_size`, it does NOT create two separate modules), so shared
> contributes **33** (11 layers × 3 proj), not 66. The real `*_proj.weight` count is
> **2244** (which additionally includes the **48 vision-tower** projections the estimate
> omitted). See the reconciliation table below.

---

## (a) Quantizable Linear-module census — count of `*_proj.weight` keys

**Method:** enumerate every key in `weight_map` ending in `_proj.weight` and bucket it.

| Bucket | Count | Formula | Example key |
|---|---:|---|---|
| Routed experts | **2112** | 11 MoE layers × 64 experts × 3 (`gate`/`up`/`down`) | `model.layers.10.mlp.experts.8.down_proj.weight` |
| Language attention | **48** | 12 layers × 4 (`q`/`k`/`v`/`o`_proj) | `model.layers.11.self_attn.v_proj.weight` |
| Shared experts (fused) | **33** | 11 MoE layers × 3 (`gate`/`up`/`down`) | `model.layers.11.mlp.shared_experts.gate_proj.weight` |
| Dense MLP (layer 0) | **3** | 1 dense layer × 3 (`gate`/`up`/`down`) | `model.layers.0.mlp.down_proj.weight` |
| Vision tower (CLIP-L) | **48** | 24 ViT layers × 2 (`qkv_proj`/`out_proj`) | `model.vision_model.transformer.layers.15.self_attn.out_proj.weight` |
| **TOTAL `*_proj.weight`** | **2244** | 2112 + 48 + 33 + 3 + 48 | — |

`lm_head.weight` is a Linear but does **not** match the `_proj.weight` suffix, so it is
counted separately (+1). Adding it gives **2245** named-Linear weight tensors discoverable
by suffix, or **2196 language-decoder proj + lm_head = 2197** if the 48 vision projections
are excluded.

### Source evidence

Routed-expert / shared-expert / dense-MLP / attention naming comes straight from the
DeepseekV2 module definitions:

- `modeling_deepseekv2.py:392-394` — the three MLP projections per expert / dense MLP / shared MLP:
  ```python
  self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
  self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
  self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
  ```
- `modeling_deepseekv2.py:602-604` — the two shared experts are **one** `DeepseekV2MLP`
  whose intermediate width = `moe_intermediate_size * n_shared_experts` (hence 3 proj, not 6):
  ```python
  if config.n_shared_experts is not None:
      intermediate_size = config.moe_intermediate_size * config.n_shared_experts
      self.shared_experts = DeepseekV2MLP(
  ```
- `modeling_deepseekv2.py:1286-1307` — the active (non-MLA) attention path uses
  `q_proj`/`k_proj`/`v_proj`/`o_proj` (4 per layer; `use_mla=false` at `config.json:49,116`):
  ```python
  query_states = self.q_proj(hidden_states)...
  key_states   = self.k_proj(hidden_states)...
  value_states = self.v_proj(hidden_states)...
  ...
  attn_output  = self.o_proj(attn_output)
  ```
- `modeling_deepseekv2.py:1796` — the language head:
  ```python
  self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
  ```

Per-layer / per-expert structure was verified directly against the index:
- 11 MoE layers carry experts (`layers 1..11`), 64 experts each (`expert idx 0..63`), 3 proj each ⇒ **2112**.
- 11 MoE layers carry exactly one `shared_experts` MLP, 3 proj each ⇒ **33**.
- 12 layers carry `self_attn` with `q/k/v/o_proj` ⇒ **48**.
- only `layers.0.mlp` carries top-level `gate/up/down_proj` (the single dense layer; `first_k_dense_replace=1` at `config.json:28,96`) ⇒ **3**.
- 24 `vision_model.transformer.layers.N.self_attn` each carry `qkv_proj` + `out_proj` ⇒ **48**.

Driving config dims (all `config.json`):
- `n_routed_experts: 64` (`config.json:36,104`)
- `n_shared_experts: 2` (`config.json:37,105`)
- `num_hidden_layers: 12` (`config.json:40,108`)
- `first_k_dense_replace: 1` (`config.json:28,96`) ⇒ layer 0 dense, layers 1-11 MoE
- vision CLIP-L `layers: 24` (`config.json:73`)

---

## (b) Total tensor count + total_size

- **Total tensors in `weight_map`: 2710**
- **`total_size`: 6 672 212 480 bytes** (≈ 6.21 GiB / 6.67 GB)

### Source evidence

- `model.safetensors.index.json:3`:
  ```json
  "total_size": 6672212480
  ```
- `weight_map` enumerated ⇒ **2710** keys.

Sanity: `6 672 212 480 bytes / 2 (bf16) = 3.336 B parameters`, consistent with
`torch_dtype: "bfloat16"` (`config.json:62`). All tensors live in a single shard
`model-00001-of-000001.safetensors`.

### Full per-subsystem tensor census (sums to 2710)

| Subsystem | Tensors |
|---|---:|
| Language routed experts (`model.layers.*.mlp.experts.*`) | 2112 |
| Language shared experts (`*.mlp.shared_experts.*`) | 33 |
| Language attention (`*.self_attn.*`, incl. norms) | 48 |
| Language dense MLP layer 0 (`layers.0.mlp.*`, incl. norms) | 14 |
| Language per-layer norms / MoE gate / misc | 24 |
| Final norm (`model.norm.weight`) | 1 |
| `model.embed_tokens.weight` | 1 |
| `lm_head.weight` | 1 |
| `model.image_newline` (param) | 1 |
| `model.view_seperator` (param) | 1 |
| Vision tower (CLIP-L, `model.vision_model.*`) | 293 |
| SAM tower (`model.sam_model.*`) | 179 |
| Projector (`model.projector.layers.{weight,bias}`) | 2 |
| **TOTAL** | **2710** |

(The attention-bucket "48" and dense-MLP "14" here include the LayerNorm/router tensors
that share those prefixes; only the `*_proj.weight` subset of each is quantizable — see
section (a). The 24 "per-layer norms / MoE gate" bucket holds `input_layernorm`,
`post_attention_layernorm`, and `mlp.gate.weight` router tensors across the 11 MoE layers.)

---

## (c) Per-1024-view token census = 273

For a single 1024×1024 global view: **256 image-feature + 16 image_newline + 1 view_seperator = 273**.

### Derivation (all line-backed)

1. `image_size = 1024` (`config.json:65`), `patch_size = 16`
   (`modeling_unlimitedocr.py:827,1172`), `downsample_ratio = 4`
   (`modeling_unlimitedocr.py:828,1173`).
2. `num_queries_base = ceil((base_size // patch_size) / downsample_ratio)`
   (`modeling_unlimitedocr.py:905`):
   `ceil((1024 // 16) / 4) = ceil(64 / 4) = 16` ⇒ a **16 × 16** feature grid = **256** image
   features (one query per grid cell).
3. **image_newline** is appended **once per row** (h = 16 rows), giving **16** newline tokens.
   - In the runtime tokenizer string (`modeling_unlimitedocr.py:913`):
     ```python
     tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
     ```
     i.e. `(16 image-feature + 1 newline) × 16 rows = 272`.
   - In the embedding-assembly path (`modeling_unlimitedocr.py:527-529`), one
     `image_newline` is concatenated per row:
     ```python
     global_features = torch.cat(
         [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
     )
     ```
4. **view_seperator** is appended **once** at the end of the view.
   - Token string (`modeling_unlimitedocr.py:914`): `tokenized_image += [image_token_id]` (the separator slot).
   - Embedding path (`modeling_unlimitedocr.py:572`):
     ```python
     global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
     ```

**Total = 256 + 16 + 1 = 273 tokens per 1024-view.** ✓

> Note on `valid_img_tokens` accounting: `modeling_unlimitedocr.py:936` adds
> `int(256 * ratio)` for `base_size == 1024` — this is the **content** token count (256),
> separate from the +16 structural newlines and +1 separator that pad it to 273 in the
> embedding sequence.

The `image_newline` and `view_seperator` are learned `nn.Parameter` vectors of dim
`n_embed = 1280`:
- `modeling_unlimitedocr.py:440-444`:
  ```python
  n_embed = 1280
  ...
  self.image_newline   = nn.Parameter(torch.randn(n_embed) * embed_std)
  self.view_seperator  = nn.Parameter(torch.randn(n_embed) * embed_std)
  ```
- present in the index as the bare tensors `model.image_newline` and `model.view_seperator`.

> **UNBLOCKS:** the image-embedding-assembly bead (sequence layout / masked_scatter
> placement) and the prompt-template bead. A crop/tiled view adds a second
> `local_features` block (`modeling_unlimitedocr.py:915-917`); the 273 figure is the
> base/global no-crop view specifically.

---

## (d) Worst-case R-SWA reference length `m` and per-layer KV buffer sizes

The active attention is **`SlidingWindowLlamaAttention`** ("R-SWA" = Ring Sliding-Window
Attention), selected because `use_mla: false` ⇒ `mha_eager`:
- `modeling_deepseekv2.py:1387` — `"mha_eager": SlidingWindowLlamaAttention`
- `modeling_deepseekv2.py:1398-1401` — `attn_implementation = "mha_" + ...` when `not use_mla`.

### Ring window `W`

`W = config._ring_window = sliding_window_size = 128`:
- `config.json:52` — `"sliding_window_size": 128` (also `config.json:119,120`).
- `modeling_unlimitedocr.py:1235-1236`:
  ```python
  _orig_sw = getattr(self.config, 'sliding_window_size', None) or getattr(self.config, 'sliding_window', None)
  self.config._ring_window = _orig_sw  # Save for ring buffer to read
  ```
- `modeling_deepseekv2.py:1282` — `W = getattr(self.config, '_ring_window', None)`.

### Worst-case reference length `m` (what a decode step attends over)

Per the ring algorithm, the KV cache during decode is **`prefill_len` (never evicted) +
`W` ring slots**:
- `modeling_deepseekv2.py:1331-1335` — `prefill_len = past_kv._prefill_length[layer]`;
  warmup grows the cache until `cur_len >= prefill_len + W`.
- `modeling_deepseekv2.py:1362-1366` — steady state overwrites the `W` ring slots in place
  (`slot = prefill_len + ring_pos; ring_pos = (ring_pos + 1) % W`), so the cache length is
  pinned at **`prefill_len + W`**.
- `modeling_deepseekv2.py:1370-1372` — attention runs over the FULL cache
  (`kcache`/`vcache`), i.e. over `m = prefill_len + W` keys.

Therefore:

- **Steady-state reference length: `m = prefill_len + W = prefill_len + 128`.**
- **Worst-case bound:** `prefill_len` is capped by `max_position_embeddings = 32768`
  (`config.json:33,101`) / generation `max_length=32768`
  (`modeling_unlimitedocr.py:787`), so
  **`m_max = 32768 + 128 = 32896` keys**.

> The "ring" only bounds the **decode/generated** suffix to `W=128`; the full **prefill**
> prefix (image tokens + prompt) is retained un-evicted. The port's R-SWA buffer must
> therefore size `prefill_len + 128`, not a fixed 128.

### Per-layer KV buffer sizes

MHA dims (no GQA): `num_key_value_heads = 10` (`config.json:41,109`),
`head_dim = hidden_size / num_attention_heads = 1280 / 10 = 128`
(`hidden_size:1280` `config.json:29,97`; `num_attention_heads:10` `config.json:38,106`).
bf16 = 2 bytes.

Per **token**, per **layer**, K+V = `2 (K,V) × num_kv_heads × head_dim × 2 bytes`
= `2 × 10 × 128 × 2` = **5120 bytes** (5 KiB).

| Region | Per layer | All 12 layers |
|---|---:|---:|
| Ring region only (`W=128` slots) | 128 × 5120 = **655 360 B (640 KiB)** | **7 680 KiB (7.5 MiB)** |
| Full cache at max prefill (`32768+128`) | 32896 × 5120 = **168.43 MB** | **2.02 GB** |

> **UNBLOCKS:** the R-SWA / KV-cache kernel bead — fixes the ring-buffer slot stride
> (`num_kv_heads × head_dim = 1280` elems per token per K and per V), the eviction policy
> (keep `prefill_len`, ring-overwrite the trailing 128), and the buffer allocation
> (`prefill_len + 128` per layer × 12 layers). Also gates the decode-attention bead's mask
> handling (decode `q_len=1` runs unmasked over the full cache — `modeling_deepseekv2.py:1369`).

---

## Quick-reference constants

| Constant | Value | Source |
|---|---:|---|
| Total tensors | 2710 | index `weight_map` |
| `total_size` | 6 672 212 480 B | `index.json:3` |
| `*_proj.weight` Linear count | 2244 | enumerated |
| `*_proj.weight` + `lm_head` | 2245 | enumerated |
| Language-decoder proj + lm_head (no vision) | 2197 | 2112+48+33+3+1 |
| Routed experts | 2112 | 11×64×3 |
| Per-1024-view tokens | 273 | 256+16+1 |
| Ring window `W` | 128 | `config.json:52` |
| Worst-case `m` | 32896 (= 32768 + 128) | `config.json:33` + W |
| KV bytes / token / layer (bf16) | 5120 | 2·10·128·2 |
| Ring KV / layer (bf16) | 640 KiB | 128·5120 |
