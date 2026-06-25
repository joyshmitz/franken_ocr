# R-SWA / Attention — Phase -1 Truth Pack OQ Answers

Source snapshots (HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`):
- `docs/truth-pack/snapshots/modeling_deepseekv2.py`
- `docs/truth-pack/snapshots/modeling_unlimitedocr.py`
- `docs/truth-pack/snapshots/config.json`
- `docs/truth-pack/snapshots/configuration_deepseek_v2.py`

All R-SWA logic lives in `SlidingWindowLlamaAttention` (`modeling_deepseekv2.py:1232-1377`). "R-SWA" = a **R**ing-buffer **S**liding-**W**indow **A**ttention: the prefill (reference) KV is kept in full and **never** evicted; only the decoded tokens occupy a fixed-size ring buffer of `W = sliding_window_size = 128` slots.

---

## OQ-1 — R-SWA reference set m: visual-only or visual+prompt? Exact mask boundary.

**Question (verbatim):** R-SWA reference set m: visual-only or visual+prompt? exact mask boundary?

**ANSWER (definitive):** The reference set `m` is **visual + prompt — i.e. the ENTIRE prefill sequence** (BOS + image tokens + prompt text, all of it), not visual-only. The boundary is `prefill_len`, which is recorded as the **full KV-cache length after the first (prefill) forward pass**, and the sliding window is applied **only to the decoded tokens that come after** `prefill_len`. The whole prefill is permanently retained.

Mechanics:
1. During prefill the model runs a **standard full causal mask** over the entire prefill sequence (no windowing): `modeling_deepseekv2.py:1719`
   ```
   attention_mask = _prepare_4d_causal_attention_mask(
       attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,)
   ```
   In `SlidingWindowLlamaAttention` the prefill path (`_is_true_prefill`) calls the plain `_attn_forward()` which applies that causal mask over the full cache: `modeling_deepseekv2.py:1315-1316`, `:1300-1302`.
2. The reference-set boundary `prefill_len` is the **post-prefill cache length** (= length of visual+prompt prefix): `modeling_deepseekv2.py:1322`
   ```
   past_kv._prefill_length[self.layer_idx] = _get_kcache(past_kv, self.layer_idx).shape[-2]
   ```
   (also recorded on first decode step at `:1329`).
3. The ring buffer overwrites slots **at and above** `prefill_len`, so prefill KV is never touched: `modeling_deepseekv2.py:1363-1364`
   ```
   slot = prefill_len + ring_pos
   kcache[:, :, slot:slot + 1, :] = key_states[:, :, t:t + 1, :]
   ```

So `m` = `{0 .. prefill_len-1}` = all visual tokens **and** all prompt tokens (and BOS). The window only bounds the *generated* tail. Note `config.sliding_window` is deliberately set to `None` before `generate()` specifically so the HF `DynamicCache` does **not** truncate the prefill tokens — comment at `modeling_unlimitedocr.py:1233-1234`:
```
# Disable config.sliding_window to prevent DynamicCache from truncating prefill tokens.
# The ring buffer in SlidingWindowLlamaAttention handles sliding window manually.
```

**UNBLOCKS:** the R-SWA attention kernel / KV-cache bead — the Rust port must treat the full prefill (visual + prompt + BOS) as a permanent, non-evicted reference prefix and apply the 128-slot window only to decoded tokens.

---

## OQ-2 — Is attention uniformly R-SWA across ALL 12 layers incl layer 0? Any full-attention layer?

**Question (verbatim):** is attention uniformly R-SWA across ALL 12 layers incl layer 0? any full-attention layer?

**ANSWER (definitive):** **Yes — uniformly R-SWA across all 12 layers, including layer 0. There is NO full-attention layer.** Layer selection depends only on `config.use_mla` (which is `false`), never on `layer_idx`.

Evidence:
1. The model builds every decoder layer identically for `layer_idx in range(num_hidden_layers)` (= 12, per `config.json:40,108`): `modeling_deepseekv2.py:1618-1622`
   ```
   self.layers = nn.ModuleList(
       [ DeepseekV2DecoderLayer(config, layer_idx)
         for layer_idx in range(config.num_hidden_layers) ])
   ```
2. Each `DeepseekV2DecoderLayer` chooses its attention class solely from `config.use_mla` (no `layer_idx` branch): `modeling_deepseekv2.py:1398-1405`
   ```
   if config.use_mla:
       attn_implementation = "mla_" + config._attn_implementation
   else:
       attn_implementation = "mha_" + config._attn_implementation
   self.self_attn = ATTENTION_CLASSES[attn_implementation](config=config, layer_idx=layer_idx)
   ```
3. `config.json:49` → `"use_mla": false`, so every layer takes the `mha_*` branch.
4. `mha_eager` maps to `SlidingWindowLlamaAttention` (the R-SWA class): `modeling_deepseekv2.py:1387`
   ```
   "mha_eager": SlidingWindowLlamaAttention,
   ```

No layer is special-cased to full attention; layer 0 gets `SlidingWindowLlamaAttention` like all others. (The only layer-0 specialness is the MLP type — `first_k_dense_replace=1` makes layer 0 a dense MLP instead of MoE, `modeling_deepseekv2.py:1407-1415` — but that does not affect attention.)

**UNBLOCKS:** attention-layer dispatch bead — the Rust port can hardcode R-SWA (128-window ring + permanent prefill prefix) for all 12 decoder layers; no per-layer full-attention exception is needed.

---

## OQ-3 — Warm-up vs ring-overwrite mask semantics during the first 128 decode steps (ring-buffer slot arithmetic).

**Question (verbatim):** warm-up vs ring-overwrite mask semantics during the first 128 decode steps — the ring-buffer slot arithmetic?

**ANSWER (definitive):** There are three decode regimes. `W = _ring_window = sliding_window_size = 128` (`modeling_deepseekv2.py:1282`, `config.json:52,119`).

**(a) Prefill:** `_attn_forward()` with the full causal mask, then `prefill_len` is recorded once: `modeling_deepseekv2.py:1315-1323`.

**(b) Warm-up (cat-append) — the first `W=128` decode steps:** while the cache length is still below `prefill_len + W`, each new token is **appended** to the cache (via the normal `past_kv.update`) — NOT overwritten — and attention runs over the whole cache so far: `modeling_deepseekv2.py:1334-1342`
```
# Warmup: cat-append until ring region is full
if cur_len < prefill_len + W:
    result = _attn_forward()
    new_len = _get_kcache(past_kv, self.layer_idx).shape[-2]
    if new_len >= prefill_len + W:
        ...
        past_kv._ring_pos[self.layer_idx] = 0
    return result
```
Boundary: warm-up holds while `cur_len < prefill_len + W` (`:1335`). `cur_len` is the live cache length (`:1332`). During warm-up the decoded tail grows from 0 up to `W` tokens, so the model attends to `prefill_len + (#decoded so far)` keys — i.e. the full prefill **plus every decoded token so far** (window not yet saturated). `_ring_pos` is initialized to 0 exactly when the ring region first fills (`new_len >= prefill_len + W` → `:1338-1341`).

**(c) Steady state (ring in-place overwrite) — decode step 129 onward:** once `cur_len >= prefill_len + W`, each new K/V overwrites a ring slot in place; the cache no longer grows. Slot arithmetic: `modeling_deepseekv2.py:1361-1367`
```
for t in range(q_len):
    slot = prefill_len + ring_pos
    kcache[:, :, slot:slot + 1, :] = key_states[:, :, t:t + 1, :]
    vcache[:, :, slot:slot + 1, :] = value_states[:, :, t:t + 1, :]
    ring_pos = (ring_pos + 1) % W
past_kv._ring_pos[self.layer_idx] = ring_pos
```
So the absolute slot index is `prefill_len + ring_pos`, with `ring_pos` cycling `0,1,…,W-1,0,…` (modulo `W`, `:1366`). The persisted ring position is carried across steps in `past_kv._ring_pos[layer_idx]` (`:1350,:1367`).

**Mask semantics:**
- Prefill: explicit 4D causal mask (`modeling_deepseekv2.py:1719`, applied at `:1300-1302`).
- Warm-up & steady-state decode (`q_len == 1`): **no causal mask is applied** — the model attends over the *entire* current cache. Model zeroes the mask for single-token decode (`modeling_deepseekv2.py:1708-1709`: `if seq_length == 1 and past_key_values_length > 0: attention_mask = None`), and the steady-state branch computes softmax over the full cache with no mask term: `modeling_deepseekv2.py:1369-1373`
  ```
  # Attention over full cache (no causal mask needed for decode q_len=1)
  k = _llama_repeat_kv(kcache, num_kv_groups)
  ...
  attn_weights = torch.matmul(query_states, k.transpose(2, 3)) / math.sqrt(head_dim)
  attn_weights = nn.functional.softmax(attn_weights, dim=-1, ...)
  ```
  Because the cache only physically holds `prefill_len + W` entries in steady state, "full cache" already *is* the windowed set (prefill prefix + the W most-recent decoded tokens). The window is enforced by physical overwrite, not by a mask.

**Net effect during the first 128 decode steps:** every decoded query (q_len=1) attends to the **entire prefill prefix + ALL decoded tokens so far** (no eviction yet); only at step 129+ does the oldest decoded token start getting overwritten (evicted) while the prefill prefix remains permanent.

**UNBLOCKS:** R-SWA ring-buffer KV-cache bead — the Rust port must implement (i) append-only warm-up for the first 128 decode tokens, (ii) modulo-`W` in-place overwrite of slots `[prefill_len, prefill_len+W)` thereafter, (iii) mask-free softmax over the physical (prefix + window) cache during decode.

---

## OQ-4 — Q/K head dim when use_mla=false and qk_nope/rope_head_dim=0 (read q_proj/k_proj shapes; confirm 128).

**Question (verbatim):** Q/K head dim when use_mla=false and qk_nope/rope_head_dim=0 — read the q_proj/k_proj shapes; confirm 128?

**ANSWER (definitive — CONFIRMED 128):** With `use_mla=false`, the DeepseekV2 MLA `q_proj`/`kv_*` projections are **not used**. The decoder uses `SlidingWindowLlamaAttention`, which subclasses the stock `transformers` `LlamaAttention` (imported at `modeling_deepseekv2.py:37-40`). Its `q_proj`/`k_proj`/`v_proj` and `head_dim` come from `LlamaAttention`, where `head_dim = hidden_size // num_attention_heads`. With `hidden_size=1280` and `num_attention_heads=10`, **head_dim = 1280 / 10 = 128**. The `qk_nope_head_dim=0` / `qk_rope_head_dim=0` config values belong to the MLA path and are irrelevant here.

Evidence:
1. `head_dim` used by the R-SWA forward is `self.head_dim` (from the Llama parent), and Q/K/V are reshaped to `(bsz, q_len, n_heads, head_dim)`: `modeling_deepseekv2.py:1278`, `:1286-1288`
   ```
   head_dim = self.head_dim
   ...
   query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
   key_states   = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
   value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
   ```
2. `config.json` has **no** `head_dim` key, so `LlamaAttention` falls back to `hidden_size // num_attention_heads`. Relevant config values:
   - `config.json:29` → `"hidden_size": 1280`
   - `config.json:38,41` → `"num_attention_heads": 10`, `"num_key_value_heads": 10` (so MHA, `num_kv_groups=1`)
   - ⇒ `head_dim = 1280 // 10 = 128`.
3. Consistency check: `config.json:50` → `"v_head_dim": 128` matches this, and `config.json:43-44` → `"qk_nope_head_dim": 0`, `"qk_rope_head_dim": 0` (MLA-only fields, unused under `use_mla=false`).

So the effective per-head dimension for Q, K, and V is **128**, with **10 query heads = 10 KV heads** (plain MHA, `num_key_value_groups=1`, `modeling_deepseekv2.py:1279`).

**Caveat (not a blocker):** the literal `q_proj`/`k_proj` weight tensors are owned by the upstream `transformers` `LlamaAttention.__init__` (not redefined in this snapshot), so their exact `out_features` are inferred from the imported class's standard formula (`num_heads * head_dim = 10*128 = 1280` for q_proj, same for k/v since num_kv_heads=10) rather than read from an explicit `nn.Linear(...)` line in the local source. This is the documented transformers-4.46.3 behavior (`config.json:63`).

**UNBLOCKS:** Q/K/V projection + RoPE kernel bead — the Rust port hardcodes head_dim=128, 10 Q heads, 10 KV heads (MHA, no GQA grouping), q_proj/k_proj/v_proj each `1280 -> 1280`, RoPE applied over the full 128-dim head via the Llama RoPE path (`_llama_apply_rotary_pos_emb`, `modeling_deepseekv2.py:1291`).

---

## OQ-13 — Does the multi-page reference block span ALL pages so page N attends to 1..N-1? (read reference/prefix KV assembly; whether sliding window excludes it.)

**Question (verbatim):** does the multi-page reference block span ALL pages so page N attends to 1..N-1? — read how the reference/prefix KV is assembled and whether sliding window excludes it?

**ANSWER (definitive — YES):** In multi-page mode, **all pages' image tokens are concatenated into a single contiguous prefill sequence in one `generate()` call**, so during prefill page N attends causally to pages 1..N-1 (full causal attention over the whole multi-page prefix). The entire multi-page prefix is the permanent reference block; the sliding window **does NOT** evict it — the 128-slot ring buffer only bounds the *decoded* tokens.

Assembly (`modeling_unlimitedocr.py`, `infer_multi`):
1. The prompt has a single `<image>` token; **all** image files go into one `images` list and are concatenated at that one position: docstring `modeling_unlimitedocr.py:1142-1144`
   ```
   All images' token sequences are concatenated at that single <image> position,
   separated by a single image_token_id between each image (same as crop mode separator).
   ```
2. The loop appends every page's image tokens (plus a one-token separator) into the **same** `tokenized_str`: `modeling_unlimitedocr.py:1198-1212`
   ```
   for idx, image in enumerate(images):
       ...
       tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
       tokenized_image += [image_token_id]  # separator token between images
       tokenized_str += tokenized_image
       images_seq_mask += [True] * len(tokenized_image)
   ```
   Then prompt text after `<image>` (`:1215-1217`) and BOS (`:1219-1222`) are prepended/appended into the **single** `input_ids`.
3. There is **one** `self.generate()` call over that single combined `input_ids` (no per-page cache reset): `modeling_unlimitedocr.py:1240-1256`
   ```
   gen_kwargs = dict(input_ids=input_ids.unsqueeze(0).cuda(), ... use_cache=True)
   ...
   output_ids = self.generate(**gen_kwargs)
   ```

Sliding window does **not** exclude the multi-page prefix:
4. As in OQ-1/OQ-3, the full concatenated prefill runs under the standard causal mask (`modeling_deepseekv2.py:1719`), and `prefill_len` is recorded as the full combined-prefix length (`modeling_deepseekv2.py:1322`). Ring overwrites only slots `>= prefill_len` (`modeling_deepseekv2.py:1363-1364`), so no page's prefill KV is ever evicted.
5. `config.sliding_window` is explicitly nulled before generation precisely so the prefill (all pages) is not truncated by `DynamicCache`: `modeling_unlimitedocr.py:1233-1237`
   ```
   # Disable config.sliding_window to prevent DynamicCache from truncating prefill tokens.
   ...
   self.config._ring_window = _orig_sw  # Save for ring buffer to read
   self.config.sliding_window = None
   ```

**Conclusion:** the reference block spans ALL pages. Within the prefill, page N's tokens attend (causally) to every token of pages 1..N-1 and the earlier portion of page N; that cross-page reference KV is permanent and the 128-window applies only to generated output tokens.

**UNBLOCKS:** multi-page / cross-page attention bead — the Rust port must build one combined prefill sequence (BOS + concatenated per-page image-token blocks with single-token separators + prompt) processed under a single full causal prefill, and treat the entire multi-page prefix as the non-evicted R-SWA reference; the sliding window must not drop any page's KV.

---

## Cross-cutting confirmed constants

| Quantity | Value | Source |
|---|---|---|
| `sliding_window_size` (= ring window `W`) | 128 | `config.json:52`, also `config.json:119-120`; read at `modeling_deepseekv2.py:1282` |
| `num_hidden_layers` | 12 | `config.json:40,108` |
| `use_mla` | false → all layers `SlidingWindowLlamaAttention` | `config.json:49`; `modeling_deepseekv2.py:1398-1405,1387` |
| head_dim (Q/K/V) | 128 (= 1280/10) | `config.json:29,38`; `modeling_deepseekv2.py:1278,1286-1288` |
| Q heads / KV heads | 10 / 10 (MHA, group=1) | `config.json:38,41`; `modeling_deepseekv2.py:1276-1279` |
| image_token_id | 128815 | `modeling_unlimitedocr.py:1181` |
| Decode mask | none (q_len=1) | `modeling_deepseekv2.py:1708-1709,1369` |
| Prefill mask | full 4D causal | `modeling_deepseekv2.py:1719` |
