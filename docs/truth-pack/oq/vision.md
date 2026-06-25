# Phase -1 Truth Pack — Vision Encoder OQ Answers

Source: pinned HF snapshot @ commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`
Files read in full:
- `docs/truth-pack/snapshots/deepencoder.py` (1057 lines — SAM-ViT-B, CLIP-L/14, neck/downsample convs, projector)
- `docs/truth-pack/snapshots/modeling_unlimitedocr.py` (concat/forward wiring, lines 430–584)
- `docs/truth-pack/snapshots/config.json` (verified vision/projector params)

All answers below are LINE-BACKED with verbatim source quotes.

---

## OQ-6 — The EXACT SAM-feature ⊕ CLIP-feature concat order/path that forms the 2048-dim projector input

**Question (verbatim):** The EXACT SAM-feature ⊕ CLIP-feature concat order/path that forms the 2048-dim projector input — quote the forward that builds it; identify `low_high_hybrid_split_mlp_gelu` vs `hybrid_split_feature_mlp_gelu` and which is used.

### ANSWER (definitive)

The 2048-dim projector input is built in `modeling_unlimitedocr.py` (NOT in `deepencoder.py`). The forward calls both encoders, then concatenates **CLIP first, SAM second**, along the channel dim (`dim=-1`):

1. SAM runs first and produces the high-res feature map; it is also passed INTO the CLIP model as `patch_embeds`:
   - `modeling_unlimitedocr.py:499` — `local_features_1 = sam_model(patches)`
   - `modeling_unlimitedocr.py:501` — `local_features_2 = vision_model(patches, local_features_1)`
2. The concat order (the load-bearing line):
   - `modeling_unlimitedocr.py:503`:
     ```python
     local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
     ```
   - Identical pattern for the global/base path at `modeling_unlimitedocr.py:509` and the no-crop path at `:558`:
     ```python
     global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
     ```

**Concat order is: `[CLIP_features (drop CLS), SAM_features]` along `dim=-1`.**
- First operand `*_features_2[:, 1:]` = CLIP (`vision_model` = `build_clip_l`) output, with the leading CLS token at index 0 dropped (`[:, 1:]`). CLIP width = 1024.
- Second operand `*_features_1.flatten(2).permute(0, 2, 1)` = SAM (`sam_model` = `build_sam_vit_b`) output. SAM `net_3` outputs 1024 channels in `[B, 1024, H, W]`; `.flatten(2)` → `[B, 1024, H*W]`; `.permute(0,2,1)` → `[B, H*W, 1024]`.
- Result channel dim = 1024 (CLIP) + 1024 (SAM) = **2048**, matching `input_dim=2048`.

**Projector identity:** The projector is a plain **`linear`** projector — NEITHER hybrid split variant is used.
- `modeling_unlimitedocr.py:441`:
  ```python
  self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
  ```
  with `n_embed = 1280` (`modeling_unlimitedocr.py:440`).
- Confirmed in `config.json:55-60` (`projector_config`): `"input_dim": 2048`, `"n_embed": 1280`, `"projector_type": "linear"`.
- The `linear` branch in the projector is just `nn.Linear(input_dim, n_embed)`:
  - `deepencoder.py:31-32`:
    ```python
    elif cfg.projector_type == "linear":
        modules = nn.Linear(cfg.input_dim, cfg.n_embed)
    ```
- Projector application: `local_features = self.projector(local_features)` (`modeling_unlimitedocr.py:504`), `global_features = self.projector(global_features)` (`:510`, `:559`).

**`low_high_hybrid_split_mlp_gelu` vs `hybrid_split_feature_mlp_gelu` (both DEFINED, NEITHER USED here):**
- `low_high_hybrid_split_mlp_gelu` (`deepencoder.py:67-76` init; `:131-135` forward): expects `x` to be a 2-element sequence (`high_x, low_x = x[0], x[1]`), runs two separate `Linear(input_dim, n_embed//2)` up-projections (`high_up_proj`, `low_up_proj`), then `torch.concat([high_x, low_x], dim=-1)`. `input_dim` is a scalar; each branch → `n_embed//2`.
- `hybrid_split_feature_mlp_gelu` (`deepencoder.py:78-88` init; `:137-142` forward): expects a SINGLE already-concatenated tensor and SPLITS it by channel using a 2-element `input_dim` list — `high_x = x[..., :input_dim[0]]`, `low_x = x[..., input_dim[0]:]` — with `high_up_proj = Linear(input_dim[0], int(n_embed*channel_div))` and `low_up_proj = Linear(input_dim[1], n_embed - int(n_embed*channel_div))` (`channel_div` default 0.5, `:80`), then `torch.concat([high_x, low_x], dim=-1)`.
- **This deployed checkpoint uses neither**: `projector_type` is `"linear"` (config + code, above). The hybrid-split modules' `high_up_proj`/`low_up_proj` weights are never constructed because the `linear` branch is taken in `MlpProjector.__init__`.

**UNBLOCKS:** Vision feature-fusion bead / projector kernel. The fusion is fixed and simple: concat `[CLIP[:,1:] , SAM_flattened]` → `[N, 2048]` → single `nn.Linear(2048, 1280)` (no GELU, no MLP depth, no channel-split). Note for the kernel: SAM features pass through SAM end-to-end (incl. `neck` + `net_2` + `net_3`), AND SAM's pre-`flatten` output (`local_features_1`, the `net_3` `[B,1024,H,W]` map) is ALSO injected into CLIP as `patch_embeds`, so the CLIP patch-embed Conv2d is bypassed and CLIP consumes SAM's downsampled grid as its patch tokens.

---

## OQ-15 — SAM windowed-attention window SIZE + windowed-block position-embedding scheme

**Question (verbatim):** SAM windowed-attention window SIZE — is it 14? — and the windowed-block position-embedding scheme; quote the block construction + `global_attn_indexes [2,5,8,11]`.

### ANSWER (definitive)

**Window size = 14. YES, it is 14.**
- `deepencoder.py:1043` (inside `_build_sam` → `ImageEncoderViT(...)`):
  ```python
  window_size=14,
  ```
- `build_sam_vit_b` sets the global-attention block indexes (`deepencoder.py:1005-1012`):
  ```python
  def build_sam_vit_b(checkpoint=None):
      return _build_sam(
          encoder_embed_dim=768,
          encoder_depth=12,
          encoder_num_heads=12,
          encoder_global_attn_indexes=[2, 5, 8, 11],
          checkpoint=checkpoint,
      )
  ```
  These `encoder_global_attn_indexes` flow into `_build_sam(...)` → `ImageEncoderViT(global_attn_indexes=encoder_global_attn_indexes, window_size=14, ...)` (`deepencoder.py:1042-1043`).

**Block construction — windowed vs global selection (`deepencoder.py:661-674`):**
```python
self.blocks = nn.ModuleList()
for i in range(depth):
    block = Block(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        act_layer=act_layer,
        use_rel_pos=use_rel_pos,
        rel_pos_zero_init=rel_pos_zero_init,
        window_size=window_size if i not in global_attn_indexes else 0,
        input_size=(img_size // patch_size, img_size // patch_size),
    )
    self.blocks.append(block)
```
So block `i` uses windowed attention with `window_size=14` UNLESS `i in {2,5,8,11}`, in which case `window_size=0` (global attention over the full `64×64` grid, since `img_size//patch_size = 1024//16 = 64`).

**Windowed-block position-embedding scheme:**

There are TWO distinct position embeddings in SAM ViT-B:

1. **Absolute position embedding (added ONCE, before any block, to the full grid — NOT per-window):**
   - Parameter shape `[1, 64, 64, 768]` (`deepencoder.py:654-659`):
     ```python
     self.pos_embed: Optional[nn.Parameter] = None
     if use_abs_pos:
         self.pos_embed = nn.Parameter(
             torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
         )
     ```
   - Added in `ImageEncoderViT.forward` BEFORE the block loop (`deepencoder.py:700-705`), with bicubic interpolation via `get_abs_pos_sam` when the runtime grid differs from `64×64`:
     ```python
     if self.pos_embed is not None:
         # x = x + self.pos_embed
         x = x + get_abs_pos_sam(self.pos_embed, x.size(1))
     for blk in self.blocks:
         x = blk(x)
     ```
   - `get_abs_pos_sam` (`deepencoder.py:548-567`): bicubic `F.interpolate(..., mode='bicubic', antialias=True, align_corners=False)` from `src_size` to `tgt_size` when they differ; identity otherwise.

2. **Decomposed RELATIVE position embeddings (per-attention, the actual windowed-block scheme):**
   - Enabled via `use_rel_pos=True` in `_build_sam` (`deepencoder.py:1041`).
   - Each `Block`'s `Attention` is sized by window for windowed blocks: `input_size=input_size if window_size == 0 else (window_size, window_size)` (`deepencoder.py:753`). So for windowed blocks the rel-pos tables are sized to the `14×14` window; for global blocks they're sized to the full `64×64` grid.
   - Rel-pos parameters (`deepencoder.py:810-817`):
     ```python
     self.use_rel_pos = use_rel_pos
     if self.use_rel_pos:
         assert input_size is not None, ...
         self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
         self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
     ```
     → windowed blocks: `2*14-1 = 27` entries per axis; global blocks: `2*64-1 = 127` entries per axis.
   - Rel-pos is added as an SDPA attn bias (`deepencoder.py:826-838`): `add_decomposed_rel_pos(...)` (`:934-968`) builds `rel_h + rel_w` and passes it as `attn_mask` to `scaled_dot_product_attention`.

**Window partition/unpartition (the windowing mechanics, `deepencoder.py:761-777`, `850-896`):**
- In `Block.forward`, when `window_size > 0`: `x, pad_hw = window_partition(x, self.window_size)` → attention → `x = window_unpartition(x, self.window_size, pad_hw, (H, W))` (`deepencoder.py:765-772`).
- `window_partition` pads to a multiple of `window_size` then reshapes into `[B*num_windows, 14, 14, C]` (`deepencoder.py:861-871`); `window_unpartition` reverses it and strips padding (`:888-896`).

**SUMMARY for OQ-15:** window_size = **14** (deepencoder.py:1043). Global-attention blocks = indexes **[2, 5, 8, 11]** (deepencoder.py:1010), where the per-block `window_size` is forced to 0 (deepencoder.py:672). Absolute pos-embed is a single `[1,64,64,768]` grid added once before the blocks with bicubic interpolation (deepencoder.py:654-659, 700-705, 548-567). Decomposed relative pos-embeds (`rel_pos_h`/`rel_pos_w`) are per-attention and sized to the window (`27` entries) for windowed blocks and to the full grid (`127` entries) for global blocks (deepencoder.py:753, 816-817).

**UNBLOCKS:** SAM windowed-attention kernel + rel-pos kernel + abs-pos (bicubic interpolation) kernel. Confirms: (a) windowed blocks partition into 14×14 tiles with their own 27-entry rel-pos tables, (b) blocks 2/5/8/11 are global (no partition, 127-entry rel-pos tables on the 64×64 grid), (c) a single absolute pos-embed is added once up front (not per window), interpolated bicubically if the grid size differs.

---

## Verified SAM / CLIP layer / width / patch params (line-backed)

### SAM-ViT-B (`sam_model = build_sam_vit_b()`, `modeling_unlimitedocr.py:437`)
| Param | Value | Source |
|---|---|---|
| `encoder_embed_dim` (width) | 768 | `deepencoder.py:1007` |
| `encoder_depth` (layers) | 12 | `deepencoder.py:1008` |
| `encoder_num_heads` | 12 | `deepencoder.py:1009` |
| `global_attn_indexes` | [2, 5, 8, 11] | `deepencoder.py:1010` |
| `window_size` | 14 | `deepencoder.py:1043` |
| `img_size` | 1024 | `deepencoder.py:1029`, `1034` |
| `vit_patch_size` | 16 | `deepencoder.py:1030`, `1039` |
| grid (img//patch) | 64×64 | derived: 1024//16 |
| `mlp_ratio` | 4 | `deepencoder.py:1036` |
| `qkv_bias` | True | `deepencoder.py:1040` |
| `use_rel_pos` | True | `deepencoder.py:1041` |
| `norm_layer` eps | 1e-6 | `deepencoder.py:1037` (`partial(LayerNorm, eps=1e-6)`) |
| `out_chans` (neck/`prompt_embed_dim`) | 256 | `deepencoder.py:1028`, `1044` |
| neck conv1 | Conv2d(768→256, k=1, bias=False) | `deepencoder.py:678-683` |
| neck LN2d | LayerNorm2d(256) | `deepencoder.py:684`, `692` |
| neck conv2 | Conv2d(256→256, k=3, pad=1, bias=False) | `deepencoder.py:685-691` |
| `net_2` (downsample) | Conv2d(256→512, k=3, **stride=2**, pad=1, bias=False) | `deepencoder.py:695` |
| `net_3` (downsample) | Conv2d(512→1024, k=3, **stride=2**, pad=1, bias=False) | `deepencoder.py:696` |
| forward returns | `x3` (net_3 output, 1024 ch) | `deepencoder.py:708-711` |

Config cross-check (`config.json:77-91`, `vision_config.width.sam_vit_b`): `width=768`, `layers=12`, `heads=12`, `global_attn_indexes=[2,5,8,11]`, `downsample_channels=[512,1024]`. Matches code exactly.

> Note: SAM's `net_2`/`net_3` each have stride 2, so the 64×64 grid after the neck is downsampled 64→32→16, giving a 16×16 SAM token grid at 1024² input. The flattened SAM map (`net_3` output, 1024 ch) is what is concatenated with CLIP and also fed to CLIP as `patch_embeds`.

### CLIP-L/14 (`vision_model = build_clip_l()`, `modeling_unlimitedocr.py:438`; cfg `vit_model_cfg` `deepencoder.py:514-532`)
| Param | Value | Source |
|---|---|---|
| `num_layers` | 24 | `deepencoder.py:515` |
| `hidden_size` (width) | 1024 | `deepencoder.py:516` |
| `num_heads` / `num_attention_heads` | 16 | `deepencoder.py:517-518` |
| `ffn_hidden_size` | 4096 | `deepencoder.py:519` |
| `image_size` | 224 | `deepencoder.py:529` |
| `patch_size` | 14 | `deepencoder.py:530` |
| `seq_length` / `max_position_embeddings` | 256 | `deepencoder.py:520-521` |
| `use_flash_attn` | False | `deepencoder.py:522` |
| `layernorm_epsilon` | 1e-5 | `deepencoder.py:527` |
| `pre_layernorm_epsilon` | 1e-5 | `deepencoder.py:528` |
| patch_embedding Conv2d | Conv2d(3→1024, k=14, stride=14, bias=False) | `deepencoder.py:252-258` |
| activation | quick_gelu = `x*sigmoid(1.702*x)` | `deepencoder.py:237-239`, `308` |
| pos-embed handling | bicubic `get_abs_pos` (interp + CLS) | `deepencoder.py:199-235`, `290` |

Config cross-check (`config.json:69-76`, `vision_config.width.clip-l-14-224`): `heads=16`, `image_size=224`, `layers=24`, `patch_size=14`, `width=1024`. Matches code exactly.

> NB: In the deployed forward, CLIP's own `patch_embedding` Conv2d is BYPASSED — `vision_model(patches, local_features_1)` passes SAM's `net_3` output as `patch_embeds`, and `CLIPVisionEmbeddings.forward` uses the provided `patch_embeds` directly (`deepencoder.py:274-283`) rather than running `self.patch_embedding(pixel_values)`. CLIP still prepends its CLS token and adds its abs-pos embedding (`deepencoder.py:286-292`); that CLS token is the one dropped by `[:, 1:]` at concat time.

---

## Cross-check: 2048 = CLIP(1024) + SAM(1024)
- CLIP output channels = `hidden_size` = 1024 (`deepencoder.py:516`).
- SAM output channels = `net_3` out = 1024 (`deepencoder.py:696`).
- Concat `dim=-1` → 1024 + 1024 = 2048 = `projector.input_dim` (`config.json:56`, `modeling_unlimitedocr.py:441`). Internally consistent.

---

## Blockers
None for OQ-6 and OQ-15. Both are fully answered from local pinned source (`deepencoder.py` + `modeling_unlimitedocr.py` + `config.json`). No external files (tokenizer.json, NVFP4 repo) were required for these two OQs.
