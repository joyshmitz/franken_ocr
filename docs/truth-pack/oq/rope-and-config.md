# Truth Pack — RoPE & Authoritative Config Record

Phase -1 "Truth Pack" open-question resolution for **franken_ocr** (pure-Rust CPU port of Baidu Unlimited-OCR).

Pinned model source: HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`, snapshotted locally at
`/Users/jemanuel/projects/franken_ocr/docs/truth-pack/snapshots/`.

Every claim below is LINE-BACKED against the real pinned source. Files cited:
- `config.json`
- `configuration_deepseek_v2.py`
- `modeling_deepseekv2.py`

---

## OQ-5 — `rope_theta` value; is YARN/NTK scaling active at the 32768 native context?

### Question (verbatim)

> OQ-5 (rope_theta value; is YARN/NTK scaling active at the 32768 native context? read rope_scaling/rope_theta and the DeepseekV2RotaryEmbedding selection).

### ANSWER (definitive)

- **`rope_theta = 10000.0`** (plain RoPE base). The value is NOT present in `config.json`, so the
  `DeepseekV2Config` default applies.
- **`rope_scaling = None`.** Also absent from `config.json`, so the default applies.
- **NO YARN / linear / dynamic-NTK scaling is active.** RoPE is vanilla (un-scaled) base-10000 across the
  full **`max_position_embeddings = 32768`** native context. No frequency rescaling, no mscale softmax
  adjustment, no NTK base interpolation is engaged.
- **Which RoPE actually runs at inference:** Because the language model sets **`use_mla: false`**
  (`config.json:49`, `config.json:116`), the decoder selects the **`mha_eager`** attention class
  (`modeling_deepseekv2.py:1398-1403`, `1387`), which is **`SlidingWindowLlamaAttention`** — a subclass of
  HF `LlamaAttention`. That path builds its rotary embedding via
  **`LlamaRotaryEmbedding(config=config)`** (`modeling_deepseekv2.py:1238-1240`) and applies it with
  `_llama_apply_rotary_pos_emb` (`modeling_deepseekv2.py:1290-1291`, `1358-1359`).
  `LlamaRotaryEmbedding` reads `config.rope_theta` (=10000.0) and `config.rope_scaling` (=None) from the
  config object. **The `DeepseekV2RotaryEmbedding` / `_init_rope()` selection in
  `DeepseekV2Attention` is the MLA-only path and is DEAD CODE for this checkpoint** (it is only reachable
  via `mla_eager` / `mla_flash_attention_2`, which require `use_mla: true`).

  Note the MLA-path `_init_rope()` confirms the same conclusion independently: with `rope_scaling is None`
  it instantiates the plain `DeepseekV2RotaryEmbedding(base=self.rope_theta)` and never any scaled variant
  (`modeling_deepseekv2.py:793-799`). So whichever attention class were chosen, the result is identical:
  un-scaled base-10000 RoPE.

### Source quotes (file:line)

**1. `rope_theta` and `rope_scaling` are NOT in `config.json`.**
The only `rope`-containing keys in `config.json` are `qk_rope_head_dim`, both `0`:
```
config.json:44     "qk_rope_head_dim": 0,
config.json:112    "qk_rope_head_dim": 0,
```
(grep for `rope` over `config.json` returns only those two lines — no `rope_theta`, no `rope_scaling`.)

**2. `DeepseekV2Config` defaults that therefore apply.**
```
configuration_deepseek_v2.py:155        rope_theta=10000.0,
configuration_deepseek_v2.py:156        rope_scaling=None,
configuration_deepseek_v2.py:199        self.rope_theta = rope_theta
configuration_deepseek_v2.py:200        self.rope_scaling = rope_scaling
```

**3. Native context length is 32768.**
```
config.json:33        "max_position_embeddings": 32768,   (language_config)
config.json:101     "max_position_embeddings": 32768,     (top-level)
```

**4. `use_mla` is false → MHA (Llama) attention path is active.**
```
config.json:49        "use_mla": false,    (language_config)
config.json:116     "use_mla": false,      (top-level)
```
```
modeling_deepseekv2.py:1380   ATTENTION_CLASSES = {
modeling_deepseekv2.py:1387       "mha_eager": SlidingWindowLlamaAttention,
modeling_deepseekv2.py:1388       # "mha_flash_attention_2": LlamaFlashAttention2
modeling_deepseekv2.py:1389   }
```
```
modeling_deepseekv2.py:1398       if config.use_mla:
modeling_deepseekv2.py:1399           attn_implementation = "mla_" + config._attn_implementation
modeling_deepseekv2.py:1400       else:
modeling_deepseekv2.py:1401           attn_implementation = "mha_" + config._attn_implementation
modeling_deepseekv2.py:1403       self.self_attn = ATTENTION_CLASSES[attn_implementation](
```

**5. The active MHA path uses `LlamaRotaryEmbedding(config=config)` (reads config.rope_theta / config.rope_scaling).**
```
modeling_deepseekv2.py:1232   class SlidingWindowLlamaAttention(LlamaAttention):
modeling_deepseekv2.py:1238       if not hasattr(self, 'rotary_emb'):
modeling_deepseekv2.py:1239           from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
modeling_deepseekv2.py:1240           self.rotary_emb = LlamaRotaryEmbedding(config=config)
```
```
modeling_deepseekv2.py:1290           cos, sin = self.rotary_emb(value_states, position_ids)
modeling_deepseekv2.py:1291           query_states, key_states = _llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

**6. Plain (un-scaled) RoPE inv_freq formula, base = rope_theta.**
The DeepseekV2 rotary (MLA path, dead here but confirms the math) builds standard inv_freq with no scaling:
```
modeling_deepseekv2.py:118   class DeepseekV2RotaryEmbedding(nn.Module):
modeling_deepseekv2.py:119       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
modeling_deepseekv2.py:125           inv_freq = 1.0 / (
modeling_deepseekv2.py:126               self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
modeling_deepseekv2.py:127           )
```

**7. `_init_rope()` picks the un-scaled embedding because `rope_scaling is None`.**
```
modeling_deepseekv2.py:793   def _init_rope(self):
modeling_deepseekv2.py:794       if self.config.rope_scaling is None:
modeling_deepseekv2.py:795           self.rotary_emb = DeepseekV2RotaryEmbedding(
modeling_deepseekv2.py:796               self.qk_rope_head_dim,
modeling_deepseekv2.py:797               max_position_embeddings=self.max_position_embeddings,
modeling_deepseekv2.py:798               base=self.rope_theta,
modeling_deepseekv2.py:799           )
```
The `linear` / `dynamic` / `yarn` branches (`modeling_deepseekv2.py:806-843`) are only entered when
`rope_scaling is not None`, which is not the case here.

**8. mscale softmax adjustment is skipped (no rope_scaling).**
```
modeling_deepseekv2.py:785       self.softmax_scale = self.q_head_dim ** (-0.5)
modeling_deepseekv2.py:786       if self.config.rope_scaling is not None:    # FALSE -> block skipped
modeling_deepseekv2.py:787           mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
```

### Caveat / not in snapshot (does not change the answer)

`LlamaRotaryEmbedding` and `LlamaAttention` themselves are imported from the external `transformers`
library (`modeling_deepseekv2.py:37-42`; `transformers_version: "4.46.3"` per `config.json:63`) and are
NOT included in the local snapshot. Their internal logic (inv_freq construction, default
`rope_type="default"` when `rope_scaling is None`, and head_dim = `hidden_size / num_attention_heads`)
is standard upstream HF code. The *inputs* it consumes — `rope_theta=10000.0`, `rope_scaling=None`,
`max_position_embeddings=32768`, `head_dim=1280/10=128` — are all fully line-backed above. The plain
DeepseekV2 RoPE math (item 6) is identical to upstream Llama's default RoPE, so the conclusion (un-scaled
base-10000 RoPE over the full 32768 context) holds regardless.

### What this UNBLOCKS

- **RoPE kernel / bead:** Implement vanilla RoPE with `theta = 10000.0`, rotary dim = full
  `head_dim = 128` (since `qk_rope_head_dim = 0` and the active path is Llama-style full-head RoPE), over
  positions `[0, 32768)`. **No** YARN long-rope tables, **no** dynamic-NTK base recompute, **no** linear
  position scaling, **no** mscale softmax factor. The Rust port can hardcode `inv_freq[i] = 1 /
  10000^(2i/128)` for `i in 0..64` and skip every scaling code path entirely.
- **Attention/softmax-scale bead:** softmax scale is plain `1/sqrt(head_dim)` (Llama path uses
  `1/math.sqrt(head_dim)`, `modeling_deepseekv2.py:1299`); no mscale multiplier.

---

## Authoritative Config Record (full transcription)

Transcribed verbatim from
`/Users/jemanuel/projects/franken_ocr/docs/truth-pack/snapshots/config.json`
(HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`). This is the canonical config record for the port.

### `language_config` (DeepseekV2 / DeepseekOCRForCausalLM) — `config.json:17-53`

| Field | Value | Line |
|---|---|---|
| `architectures` | `["DeepseekOCRForCausalLM"]` | config.json:18-20 |
| `auto_map.AutoConfig` | `"configuration_deepseekv2.DeepseekV2Config"` | config.json:22 |
| `auto_map.AutoModel` | `"modeling_deepseek.DeepseekV2Model"` | config.json:23 |
| `auto_map.AutoModelForCausalLM` | `"modeling_deepseek.DeepseekV2ForCausalLM"` | config.json:24 |
| `bos_token_id` | `0` | config.json:26 |
| `eos_token_id` | `1` | config.json:27 |
| `first_k_dense_replace` | `1` | config.json:28 |
| `hidden_size` | `1280` | config.json:29 |
| `intermediate_size` | `6848` | config.json:30 |
| `kv_lora_rank` | `null` | config.json:31 |
| `lm_head` | `true` | config.json:32 |
| `max_position_embeddings` | `32768` | config.json:33 |
| `moe_intermediate_size` | `896` | config.json:34 |
| `n_group` | `1` | config.json:35 |
| `n_routed_experts` | `64` | config.json:36 |
| `n_shared_experts` | `2` | config.json:37 |
| `num_attention_heads` | `10` | config.json:38 |
| `num_experts_per_tok` | `6` | config.json:39 |
| `num_hidden_layers` | `12` | config.json:40 |
| `num_key_value_heads` | `10` | config.json:41 |
| `q_lora_rank` | `null` | config.json:42 |
| `qk_nope_head_dim` | `0` | config.json:43 |
| `qk_rope_head_dim` | `0` | config.json:44 |
| `rm_head` | `false` | config.json:45 |
| `topk_group` | `1` | config.json:46 |
| `topk_method` | `"greedy"` | config.json:47 |
| `torch_dtype` | `"bfloat16"` | config.json:48 |
| `use_mla` | `false` | config.json:49 |
| `v_head_dim` | `128` | config.json:50 |
| `vocab_size` | `129280` | config.json:51 |
| `sliding_window_size` | `128` | config.json:52 |

**Defaulted (NOT in config.json, so `DeepseekV2Config.__init__` defaults apply):**
| Field | Default value | Default source line |
|---|---|---|
| `rope_theta` | `10000.0` | configuration_deepseek_v2.py:155 |
| `rope_scaling` | `None` | configuration_deepseek_v2.py:156 |
| `routed_scaling_factor` | `1.0` | configuration_deepseek_v2.py:129 |
| `moe_layer_freq` | `1` | configuration_deepseek_v2.py:139 |
| `norm_topk_prob` | `False` | configuration_deepseek_v2.py:141 |
| `scoring_func` | `'softmax'` | configuration_deepseek_v2.py:142 |
| `hidden_act` | `"silu"` | configuration_deepseek_v2.py:145 |
| `rms_norm_eps` | `1e-6` | configuration_deepseek_v2.py:148 |
| `attention_bias` | `False` | configuration_deepseek_v2.py:157 |
| `attention_dropout` | `0.0` | configuration_deepseek_v2.py:158 |
| `aux_loss_alpha` | `0.001` | configuration_deepseek_v2.py:143 |
| `seq_aux` | `True` | configuration_deepseek_v2.py:144 |
| `ep_size` | `1` | configuration_deepseek_v2.py:128 |
| `initializer_range` | `0.02` | configuration_deepseek_v2.py:147 |
| `use_cache` | `True` | configuration_deepseek_v2.py:149 |
| `pretraining_tp` | `1` | configuration_deepseek_v2.py:153 |
| `tie_word_embeddings` | `False` | configuration_deepseek_v2.py:154 |
| `model_type` | `"deepseek_v2"` (class attr) | configuration_deepseek_v2.py:114 |

> NOTE on `sliding_window`: the **language_config** uses key `sliding_window_size` (=128, config.json:52),
> NOT the `DeepseekV2Config.__init__` param `sliding_window` (which defaults to `None`,
> configuration_deepseek_v2.py:160). At the **top level** of config.json both `sliding_window_size: 128`
> (config.json:119) AND `sliding_window: 128` (config.json:120) are present. So the effective sliding
> window is **128** and is plumbed in via the top-level / `_ring_window` mechanism, not via the
> DeepseekV2Config default. (The MHA attention reads `getattr(config, 'sliding_window', None)` at
> modeling_deepseekv2.py:1243 and `_ring_window` at modeling_deepseekv2.py:1282.)

### Top-level config — `config.json:1-16, 54-121`

| Field | Value | Line |
|---|---|---|
| `_name_or_path` | `"Unlimited-OCR"` | config.json:2 |
| `candidate_resolutions` | `[[1024, 1024]]` | config.json:3-8 |
| `global_view_pos` | `"head"` | config.json:9 |
| `architectures` | `["UnlimitedOCRForCausalLM"]` | config.json:10-12 |
| `auto_map.AutoConfig` | `"modeling_unlimitedocr.UnlimitedOCRConfig"` | config.json:14 |
| `auto_map.AutoModel` | `"modeling_unlimitedocr.UnlimitedOCRForCausalLM"` | config.json:15 |
| `model_type` | `"unlimited-ocr"` | config.json:54 |
| `tile_tag` | `"2D"` | config.json:61 |
| `torch_dtype` | `"bfloat16"` | config.json:62 |
| `transformers_version` | `"4.46.3"` | config.json:63 |
| `bos_token_id` | `0` | config.json:94 |
| `eos_token_id` | `1` | config.json:95 |
| `first_k_dense_replace` | `1` | config.json:96 |
| `hidden_size` | `1280` | config.json:97 |
| `intermediate_size` | `6848` | config.json:98 |
| `kv_lora_rank` | `null` | config.json:99 |
| `lm_head` | `true` | config.json:100 |
| `max_position_embeddings` | `32768` | config.json:101 |
| `moe_intermediate_size` | `896` | config.json:102 |
| `n_group` | `1` | config.json:103 |
| `n_routed_experts` | `64` | config.json:104 |
| `n_shared_experts` | `2` | config.json:105 |
| `num_attention_heads` | `10` | config.json:106 |
| `num_experts_per_tok` | `6` | config.json:107 |
| `num_hidden_layers` | `12` | config.json:108 |
| `num_key_value_heads` | `10` | config.json:109 |
| `q_lora_rank` | `null` | config.json:110 |
| `qk_nope_head_dim` | `0` | config.json:111 |
| `qk_rope_head_dim` | `0` | config.json:112 |
| `rm_head` | `false` | config.json:113 |
| `topk_group` | `1` | config.json:114 |
| `topk_method` | `"greedy"` | config.json:115 |
| `use_mla` | `false` | config.json:116 |
| `v_head_dim` | `128` | config.json:117 |
| `vocab_size` | `129280` | config.json:118 |
| `sliding_window_size` | `128` | config.json:119 |
| `sliding_window` | `128` | config.json:120 |

### `projector_config` — `config.json:55-60`

| Field | Value | Line |
|---|---|---|
| `input_dim` | `2048` | config.json:56 |
| `model_type` | `"mlp_projector"` | config.json:57 |
| `n_embed` | `1280` | config.json:58 |
| `projector_type` | `"linear"` | config.json:59 |

### `vision_config` — `config.json:64-93`

| Field | Value | Line |
|---|---|---|
| `image_size` | `1024` | config.json:65 |
| `mlp_ratio` | `3.7362` | config.json:66 |
| `model_name` | `"deeplip_b_l"` | config.json:67 |
| `model_type` | `"vision"` | config.json:68 |
| `width.clip-l-14-224.heads` | `16` | config.json:71 |
| `width.clip-l-14-224.image_size` | `224` | config.json:72 |
| `width.clip-l-14-224.layers` | `24` | config.json:73 |
| `width.clip-l-14-224.patch_size` | `14` | config.json:74 |
| `width.clip-l-14-224.width` | `1024` | config.json:75 |
| `width.sam_vit_b.downsample_channels` | `[512, 1024]` | config.json:78-81 |
| `width.sam_vit_b.global_attn_indexes` | `[2, 5, 8, 11]` | config.json:82-87 |
| `width.sam_vit_b.heads` | `12` | config.json:88 |
| `width.sam_vit_b.layers` | `12` | config.json:89 |
| `width.sam_vit_b.width` | `768` | config.json:90 |

### Derived constants for the language tower (line-backed)

- `head_dim` (MHA/Llama path) = `hidden_size / num_attention_heads` = `1280 / 10` = **128**.
  (`num_attention_heads=10` at config.json:106; `hidden_size=1280` at config.json:97. The `head_dim`
  division itself lives in the external HF `LlamaAttention` base class — see caveat under OQ-5.)
  Note `qk_rope_head_dim=0` and `qk_nope_head_dim=0` (config.json:111-112) belong to the MLA path, which
  is inactive; the active Llama path derives head_dim from hidden_size/num_heads.
- `v_head_dim = 128` (config.json:117) — matches head_dim, consistent with full-head MHA.
- MoE: `n_routed_experts=64`, `num_experts_per_tok=6`, `n_shared_experts=2`,
  `moe_intermediate_size=896`, `first_k_dense_replace=1`, `n_group=1`, `topk_group=1`,
  `topk_method="greedy"` (config.json:104-107, 102, 96, 103, 114-115).
