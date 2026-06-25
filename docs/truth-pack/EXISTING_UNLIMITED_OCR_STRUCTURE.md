# EXISTING_UNLIMITED_OCR_STRUCTURE.md

**Bead:** PM1-spec-extraction — THE structural spec the Rust port (`franken_ocr`) implements from.

**Pinned source:** Baidu Unlimited-OCR @ HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`
(local snapshots under `docs/truth-pack/snapshots/`).

**Method:** Per `/porting-to-rust` (Essence Extraction): extract spec from the ACTUAL pinned
source -> implement Rust from this spec. Every `[SPEC-NNN]` clause below is line-backed to the
real source file. `file:line` citations use the snapshot file basenames:
- `config.json`
- `processor_config.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `configuration_deepseek_v2.py`
- `conversation.py`
- `modeling_unlimitedocr.py`
- `modeling_deepseekv2.py`
- `deepencoder.py`
- `model.safetensors.index.json`

Where the source genuinely does not resolve a detail, it is marked **[UNRESOLVED]** with what is
still needed (an OQ or a missing file).

---

## 0. TOP-LEVEL ARCHITECTURE

**[SPEC-001] Model class hierarchy.** `UnlimitedOCRForCausalLM` (architecture in config) subclasses
`DeepseekV2ForCausalLM`; its `.model` is a `UnlimitedOCRModel` which subclasses `DeepseekV2Model`
and adds the vision tower. The LM head is a separate `nn.Linear(hidden_size, vocab_size, bias=False)`.

> `config.json:10-11` `"architectures": ["UnlimitedOCRForCausalLM"]`
> `modeling_unlimitedocr.py:595` `class UnlimitedOCRForCausalLM(DeepseekV2ForCausalLM):`
> `modeling_unlimitedocr.py:602` `self.model = UnlimitedOCRModel(config)`
> `modeling_unlimitedocr.py:606` `self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)`
> `modeling_unlimitedocr.py:431-432` `class UnlimitedOCRModel(DeepseekV2Model):`

**[SPEC-002] Config class is DeepseekV2Config subclass** with `model_type="unlimited-ocr"`; all
decoder hyperparameters are inherited from `DeepseekV2Config`.

> `modeling_unlimitedocr.py:428-429` `class UnlimitedOCRConfig(DeepseekV2Config): model_type = "unlimited-ocr"`

**[SPEC-003] Weight key prefixes (for safetensors load mapping).** Vision/connector weights live
under `model.sam_model.*`, `model.vision_model.*`, `model.projector.layers.{weight,bias}`,
`model.image_newline`, `model.view_seperator`; decoder under `model.layers.*`, `model.embed_tokens`,
`model.norm`; head under `lm_head.weight`. Single shard `model-00001-of-000001.safetensors`.

> `model.safetensors.index.json:9` `"model.sam_model.blocks.3.norm2.weight"`
> `model.safetensors.index.json:8` `"model.vision_model.transformer.layers.17.mlp.fc2.bias"`
> `model.safetensors.index.json:134` `"model.projector.layers.bias"`
> `model.safetensors.index.json:98` `"model.vision_model.embeddings.class_embedding"`

---

## 1. CONFIG / DATA STRUCTURES

### 1.1 Decoder config (`config.json` top-level + `language_config`)

**[SPEC-010] Core decoder dims.**
> `config.json:97` `"hidden_size": 1280`
> `config.json:98` `"intermediate_size": 6848`  (dense MLP)
> `config.json:108` `"num_hidden_layers": 12`
> `config.json:106` `"num_attention_heads": 10`
> `config.json:109` `"num_key_value_heads": 10`  (=> MHA, no GQA; `n_rep=1`)
> `config.json:118` `"vocab_size": 129280`
> `config.json:101` `"max_position_embeddings": 32768`
> `config.json:50` `"v_head_dim": 128`

**[SPEC-011] head_dim derivation (MHA path).** The runtime attention is
`SlidingWindowLlamaAttention` (LlamaAttention), so `head_dim = hidden_size / num_attention_heads
= 1280 / 10 = 128`. (This is the Llama `self.head_dim`, NOT `v_head_dim` or qk_*_head_dim, which are
MLA-only and set to 0 here.) `use_mla=False` selects the MHA branch.
> `config.json:116` `"use_mla": false`
> `config.json:43-44` `"qk_nope_head_dim": 0`, `"qk_rope_head_dim": 0` (MLA dims unused)
> `modeling_deepseekv2.py:1278` `head_dim = self.head_dim` (from LlamaAttention; = hidden/num_heads)
> `modeling_deepseekv2.py:1299` `... / math.sqrt(head_dim)` (attention scale = 1/sqrt(128))

**[SPEC-012] MoE config.**
> `config.json:105` `"n_shared_experts": 2`
> `config.json:104` `"n_routed_experts": 64`
> `config.json:107` `"num_experts_per_tok": 6` (top-k routed)
> `config.json:102` `"moe_intermediate_size": 896`
> `config.json:96` `"first_k_dense_replace": 1` (layer 0 is dense MLP, layers 1..11 are MoE)
> `config.json:114` `"topk_method": "greedy"`
> `config.json:103` `"n_group": 1`, `config.json:115` `"topk_group": 1`

**[SPEC-013] Defaults NOT in config.json — inherited from `DeepseekV2Config.__init__`.** These
constructor defaults apply because config.json does not override them:
> `configuration_deepseek_v2.py:155` `rope_theta=10000.0`
> `configuration_deepseek_v2.py:148,196` `rms_norm_eps=1e-6` (stored `float(rms_norm_eps)`)
> `configuration_deepseek_v2.py:145` `hidden_act="silu"`
> `configuration_deepseek_v2.py:139` `moe_layer_freq=1`
> `configuration_deepseek_v2.py:129` `routed_scaling_factor=1.0`
> `configuration_deepseek_v2.py:142` `scoring_func='softmax'`
> `configuration_deepseek_v2.py:141` `norm_topk_prob=False`
> `configuration_deepseek_v2.py:157` `attention_bias=False`
> `configuration_deepseek_v2.py:158` `attention_dropout=0.0`
> `configuration_deepseek_v2.py:156` `rope_scaling=None` (=> plain `DeepseekV2RotaryEmbedding`)
> `configuration_deepseek_v2.py:154` `tie_word_embeddings=False`
>
> NOTE on `norm_topk_prob`: with `num_experts_per_tok=6 (>1)` and `norm_topk_prob=False`, the gate
> takes the `else` branch: `topk_weight = topk_weight * routed_scaling_factor`
> (`modeling_deepseekv2.py:506-507`). With `routed_scaling_factor=1.0` weights are the raw softmax
> top-k probs (NOT renormalized to sum 1).

**[SPEC-014] BOS/EOS/PAD.**
> `config.json:94` `"bos_token_id": 0`; `config.json:95` `"eos_token_id": 1`
> `special_tokens_map.json:18-31` bos=`<｜begin▁of▁sentence｜>`, eos=`<｜end▁of▁sentence｜>`
> `tokenizer_config.json:6-7` id `0` = `<｜begin▁of▁sentence｜>`; `tokenizer_config.json:14-15` id `1` = `<｜end▁of▁sentence｜>`
> `tokenizer_config.json:6657` `"pad_token": "<｜▁pad▁｜>"`
> `processor_config.json:24` processor `pad_token` = `<｜▁pad▁｜>`

**[SPEC-015] Sliding window config.** Two keys both = 128. The ring-buffer reads `sliding_window_size`
(preferred) else `sliding_window`; it is saved to `config._ring_window` and `config.sliding_window`
is set to `None` during `generate` so the HF `DynamicCache` does NOT truncate prefill tokens.
> `config.json:52` `"sliding_window_size": 128`
> `config.json:119-120` `"sliding_window_size": 128`, `"sliding_window": 128`
> `modeling_unlimitedocr.py:998-1000` `_orig_sw = getattr(..., 'sliding_window_size', None) or getattr(..., 'sliding_window', None); self.config._ring_window = _orig_sw; self.config.sliding_window = None`

### 1.2 Projector / vision config

**[SPEC-016] Projector.** `projector_type="linear"`, `input_dim=2048`, `n_embed=1280`.
> `config.json:55-60` `"projector_config": {"input_dim": 2048, "n_embed": 1280, "projector_type": "linear"}`
> `modeling_unlimitedocr.py:441` `self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))` with `n_embed=1280` (`:440`)

**[SPEC-017] Vision config.** SAM ViT-B (width 768, depth 12, 12 heads, global attn at [2,5,8,11]),
CLIP-L-14-224 (width 1024, 24 layers, 16 heads, patch 14). Image size 1024. `tile_tag="2D"`,
`global_view_pos="head"`, `candidate_resolutions=[[1024,1024]]`.
> `config.json:64-92` `vision_config` block
> `config.json:61` `"tile_tag": "2D"`; `config.json:9` `"global_view_pos": "head"`
> `config.json:3-8` `"candidate_resolutions": [[1024,1024]]`

### 1.3 Processor config

**[SPEC-018] Image normalization + processor.**
> `processor_config.json:11-20` `image_mean=[0.5,0.5,0.5]`, `image_std=[0.5,0.5,0.5]`, `normalize=true`
> `processor_config.json:25` `patch_size=16`; `processor_config.json:9` `downsample_ratio=4`
> `processor_config.json:21` `image_token="<image>"`; `processor_config.json:10` `ignore_id=-100`
> `processor_config.json:2` `add_special_token=false`

### 1.4 Special token IDs (postprocess + prompt)

**[SPEC-019] Token IDs.**
> `tokenizer_config.json:6550-6551` `128815` = `<image>`
> `tokenizer_config.json:6558-6559` `128816` = `<|ref|>`
> `tokenizer_config.json:6566-6567` `128817` = `<|/ref|>`
> `tokenizer_config.json:6574-6575` `128818` = `<|det|>`
> `tokenizer_config.json:6582-6583` `128819` = `<|/det|>`
> `tokenizer_config.json:6590-6591` `128820` = `<|grounding|>`
> `tokenizer_config.json:6630-6631` `128825` = `<|User|>`; `tokenizer_config.json:6638-6639` `128826` = `<|Assistant|>`
> Runtime hardcodes `image_token_id = 128815` (`modeling_unlimitedocr.py:845`, also `:1181`).

---

## 2. PREPROCESSING / TILING (the `infer` data pipeline)

All in `modeling_unlimitedocr.py`. Entry `infer(...)` defaults: `base_size=1024`, `image_size=640`,
`crop_mode=True`, `temperature=0.0`, `max_length=32768`, `no_repeat_ngram_size=0`, `ngram_window=0`.
> `modeling_unlimitedocr.py:787` signature.

**[SPEC-020] Image load.** RGB; EXIF-transpose applied on load.
> `modeling_unlimitedocr.py:27-34` `load_image`: `ImageOps.exif_transpose(image)`
> `modeling_unlimitedocr.py:302-303` `pil_img = load_image(...); pil_img.convert("RGB")`

**[SPEC-021] Normalize transform = ToTensor then Normalize(mean=0.5, std=0.5)** => maps [0,1] -> [-1,1].
> `modeling_unlimitedocr.py:332-340` `transforms.ToTensor()` then `transforms.Normalize(mean, std)`
> `modeling_unlimitedocr.py:841` `BasicImageTransform(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), normalize=True)`
> Tensors cast to bfloat16 (`:886`, `:899`, `:933`).

**[SPEC-022] Pad-to-square with mean color (the "127" pad).** Global view is padded (aspect-preserving)
to a square via `ImageOps.pad`, fill = `int(0.5*255)=127` per channel.
> `modeling_unlimitedocr.py:872-873` `global_view = ImageOps.pad(image, (base_size, base_size), color=tuple(int(x*255) for x in image_transform.mean))`
> (mean=0.5 => fill `(127,127,127)`.) Non-crop branch: `:931-932` same pad to `(image_size,image_size)`.

**[SPEC-023] Crop decision (`crop_mode=True`).** If image is <=640 in BOTH dims, `crop_ratio=[1,1]`
(no local tiling). Otherwise `dynamic_preprocess` computes a tiling grid.
> `modeling_unlimitedocr.py:859-868`:
> `if image.size[0] <= 640 and image.size[1] <= 640: crop_ratio=[1,1]` else `images_crop_raw, crop_ratio = dynamic_preprocess(image)`

**[SPEC-024] `dynamic_preprocess` tiling.** Defaults `min_num=2, max_num=32, image_size=640,
use_thumbnail=False`. Builds candidate `(i,j)` ratios with `min_num <= i*j <= max_num`, picks the
closest aspect ratio (tie-break favors larger area), resizes to `(640*i, 640*j)`, then crops a
row-major grid of 640x640 tiles. Returns tiles + `target_aspect_ratio=(width_crop_num, height_crop_num)`.
> `modeling_unlimitedocr.py:175-213` full function
> `:180-182` target_ratios set; `:184` sorted by `i*j`
> `:197` `resized_img = image.resize((target_width, target_height))`
> `:199-208` row-major crop loop (`box` uses `i % (target_width//image_size)` for column)
> `:209` `assert len(processed_images) == blocks`
> Tie-break: `:168-170` `if area > 0.5*image_size*image_size*ratio[0]*ratio[1]: best_ratio = ratio`

**[SPEC-025] `find_closest_aspect_ratio`.** Minimizes `|aspect_ratio - i/j|`.
> `modeling_unlimitedocr.py:158-172`

**[SPEC-026] crop_ratio is (width_crop_num, height_crop_num).** Local tiles are produced only when
`width_crop_num > 1 OR height_crop_num > 1`.
> `modeling_unlimitedocr.py:890` `width_crop_num, height_crop_num = crop_ratio`
> `:892` `images_spatial_crop.append([width_crop_num, height_crop_num])`
> `:895-899` local-view tensors built when `width_crop_num>1 or height_crop_num>1`

**[SPEC-027] Token query counts.**
`num_queries = ceil((image_size//patch_size)/downsample_ratio)` with `patch_size=16`, `downsample_ratio=4`.
- base/global (1024): `num_queries_base = ceil((1024//16)/4) = ceil(64/4) = 16`.
- local tile (640): `num_queries = ceil((640//16)/4) = ceil(40/4) = 10`.
> `modeling_unlimitedocr.py:827-828` `patch_size=16; downsample_ratio=4`
> `:904` `num_queries = math.ceil((image_size // patch_size) / downsample_ratio)`
> `:905` `num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)`

**[SPEC-028] Image-token id-stream layout (crop_mode, the "2D" tile_tag layout).** The placeholder
`<image>` token stream is built as:
1. Global block: `(num_queries_base * [id] + [id]) * num_queries_base` then `+ [id]`.
   I.e. for each of `num_queries_base` rows: `num_queries_base` image tokens + 1 newline token; then
   one trailing separator. (16x16 grid + 16 row-newlines + 1 view-separator.)
2. If tiled (`width_crop_num>1 or height_crop_num>1`): local block
   `(num_queries*width_crop_num * [id] + [id]) * (num_queries*height_crop_num)`.
> `modeling_unlimitedocr.py:913-914`
> `tokenized_image = ([image_token_id]*num_queries_base + [image_token_id]) * num_queries_base`
> `tokenized_image += [image_token_id]`
> `:915-917` local block appended when tiled
> `images_seq_mask += [True]*len(tokenized_image)` (`:919`)
>
> All these id positions are `image_token_id=128815`; the actual distinction between "patch token",
> "row newline" and "view separator" is positional and is resolved by the connector's
> `masked_scatter_` order (see SPEC-040+). The count must match the feature count exactly.

**[SPEC-029] Non-crop branch (`crop_mode=False`).** Resize to `(image_size, image_size)` if
`image_size<=640`, pad to square, single global block only:
`(num_queries*[id] + [id]) * num_queries + [id]`, `crop=[1,1]`.
> `modeling_unlimitedocr.py:922-957`

**[SPEC-030] BOS prepend + masks.** After all splits, prepend `bos_id=0`; `images_seq_mask` gets a
leading `False`. `input_ids` = LongTensor; `images_seq_mask` = bool tensor.
> `modeling_unlimitedocr.py:966-978` `bos_id=0; tokenized_str=[bos_id]+tokenized_str; images_seq_mask=[False]+images_seq_mask`

**[SPEC-031] Image tensor packing.** `images_ori = stack(global views)` shape `[N,3,base,base]`;
`images_crop = stack(local tiles)` shape `[P,3,?,?]` (else zeros `[1,3,base,base]`);
`images_spatial_crop` LongTensor `[N,2]`. Passed to generate as
`images=[(images_crop, images_ori)]` (a one-element list; index 0 = crop/patches, index 1 = ori).
> `modeling_unlimitedocr.py:981-992` packing; `:1004` `images=[(images_crop.cuda(), images_ori.cuda())]`

**[SPEC-032] valid_img_tokens accounting (compression-ratio metric only, not used in forward).**
> `modeling_unlimitedocr.py:836-838` `ratio = 1 - ((max(w,h)-min(w,h))/max(w,h))`
> `:875-878` base_size==1024 -> `+= int(256*ratio)`; ==1280 -> `+= int(400*ratio)`
> `:901-902` `if image_size==640: valid_img_tokens += len(images_crop_list)*100`

**[SPEC-033] Multi-image path (`infer_multi`) — no crop mode.** Single `<image>` token in the prompt;
each image resized to `(image_size,image_size)` (if `<=640`), padded, global block per image with a
single separator token between images. `images=[(dummy_crop, images_ori)]` with `dummy_crop=zeros`
(triggers the no-crop connector branch).
> `modeling_unlimitedocr.py:1139-1257`; `:1209-1210` per-image block + separator;
> `:1230,1242` `dummy_crop = torch.zeros(...)`

---

## 3. PROMPT FORMATTING

**[SPEC-034] Conversation -> prompt uses the `plain` template (`sft_format='plain'`).** PLAIN style:
concatenates message contents with empty `sep`/`sep2`. The User content (which contains the literal
`<image>`) and the empty Assistant content are concatenated; `.strip()` applied.
> `modeling_unlimitedocr.py:825` `prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')`
> `modeling_unlimitedocr.py:233-256` `format_messages` -> `get_conv_template('plain')` -> `get_prompt().strip()`
> `conversation.py:227-241` `plain` template: `system_template=""`, `roles=("","")`, `sep=""`, `sep2=""`, `sep_style=PLAIN`
> `conversation.py:75-88` PLAIN `get_prompt` concatenates `message + seps[i%2]`

**[SPEC-035] Prompt split on `<image>`.** `prompt.split('<image>')`; text segments tokenized with
`add_special_tokens=False` and `bos=False, eos=False`; image placeholders inserted at split points.
> `modeling_unlimitedocr.py:844-846` `image_token='<image>'; image_token_id=128815; text_splits = prompt.split(image_token)`
> `modeling_unlimitedocr.py:259-268` `text_encode`: `tokenizer.encode(text, add_special_tokens=False)`, optional bos=id0/eos=id1

**[SPEC-036] Roles `<|User|>`/`<|Assistant|>`.** The conversation dict uses role strings `<|User|>`
and `<|Assistant|>`, but because the `plain` template has empty roles/seps they do not appear in the
final prompt text (only the content does).
> `modeling_unlimitedocr.py:794-806` conversation construction (`role: "<|User|>"`, content=prompt, images=[file])

---

## 4. VISION TOWER (SAM + CLIP + neck + projector) FORWARD

In `deepencoder.py` (modules) and `modeling_unlimitedocr.py` (orchestration).

### 4.1 SAM ViT-B encoder (`build_sam_vit_b` -> `ImageEncoderViT`)

**[SPEC-040] SAM build params.** embed_dim=768, depth=12, num_heads=12, img_size=1024, patch_size=16,
mlp_ratio=4, out_chans=256, qkv_bias=True, use_rel_pos=True, window_size=14,
global_attn_indexes=[2,5,8,11], norm_layer=LayerNorm(eps=1e-6).
> `deepencoder.py:1005-1012` `build_sam_vit_b` -> `_build_sam(768,12,12,[2,5,8,11])`
> `deepencoder.py:1021-1045` `_build_sam`: `prompt_embed_dim=256`, `image_size=1024`, `vit_patch_size=16`, `window_size=14`, `out_chans=prompt_embed_dim`

**[SPEC-041] SAM patch embed.** `PatchEmbed` Conv2d(3->768, kernel 16, stride 16); output permuted to
`B,H,W,C` (so H=W=1024/16=64).
> `deepencoder.py:971-1002` `PatchEmbed`; `:998-1001` `proj` then `permute(0,2,3,1)`
> `deepencoder.py:647-652` `self.patch_embed = PatchEmbed(...)`

**[SPEC-042] SAM abs pos embed.** Learned `pos_embed` shape `(1, 64, 64, 768)`; added after
bicubic-interpolating (`get_abs_pos_sam`) to current grid size if mismatched.
> `deepencoder.py:654-659` `pos_embed = nn.Parameter(zeros(1, img//patch, img//patch, embed_dim))`
> `deepencoder.py:700-702` `x = x + get_abs_pos_sam(self.pos_embed, x.size(1))`
> `deepencoder.py:548-567` `get_abs_pos_sam` bicubic interpolate (antialias=True, align_corners=False)

**[SPEC-043] SAM blocks.** 12 `Block`s; block `i` uses window attention `window_size=14` UNLESS
`i in global_attn_indexes [2,5,8,11]` (then `window_size=0` => global attention).
> `deepencoder.py:661-675` loop; `:672` `window_size=window_size if i not in global_attn_indexes else 0`
> `deepencoder.py:714-777` `Block.forward`: norm1 -> (window_partition) -> attn -> (window_unpartition) -> residual; then `x + mlp(norm2(x))`
> `deepencoder.py:761-777` exact residual order

**[SPEC-044] SAM attention with decomposed relative position.** Multi-head; `scale=head_dim**-0.5`;
`use_rel_pos=True` adds `add_decomposed_rel_pos` bias to SDPA `attn_mask`.
> `deepencoder.py:780-847` `Attention`; `:805` scale; `:826-838` rel-pos bias path into `scaled_dot_product_attention(attn_mask=attn_bias)`
> `deepencoder.py:899-968` `get_rel_pos` / `add_decomposed_rel_pos`

**[SPEC-045] SAM window partition/unpartition** with padding to a multiple of `window_size`.
> `deepencoder.py:850-896`

**[SPEC-046] SAM neck + downsample to 1024 channels.** neck = Conv2d(768->256,k1) -> LayerNorm2d ->
Conv2d(256->256,k3,pad1) -> LayerNorm2d. Then `net_2`=Conv2d(256->512,k3,s2,pad1),
`net_3`=Conv2d(512->1024,k3,s2,pad1). `forward` returns `x3` (the 1024-channel feature, shape
`B,1024,16,16` for a 1024 input: 64 -> neck keeps 64 -> /2 -> 32 -> /2 -> 16).
> `deepencoder.py:677-696` neck/net_2/net_3 defs
> `deepencoder.py:698-711` forward: `x = neck(x.permute(0,3,1,2)); x2 = net_2(x); x3 = net_3(x2.clone()); return x3`
> `deepencoder.py:590-602` `LayerNorm2d`

### 4.2 CLIP-L vision (`build_clip_l` -> `VitModel`)

**[SPEC-047] CLIP build params** (`vit_model_cfg`). num_layers=24, hidden_size=1024,
num_attention_heads=16, ffn_hidden_size=4096, image_size=224, patch_size=14, use_flash_attn=False,
layernorm_epsilon=1e-5, pre_layernorm_epsilon=1e-5.
> `deepencoder.py:514-539` `vit_model_cfg` + `build_clip_l`

**[SPEC-048] CLIP embeddings take SAM features as patch_embeds (fused tower).** `CLIPVisionEmbeddings.forward(pixel_values, patch_embeds)`:
if `patch_embeds is not None` it is used directly (i.e. the SAM output `x3` is fed in as the patch
embedding), flattened `flatten(2).transpose(1,2)`; class token prepended; abs pos added via
`get_abs_pos` (bicubic interp to current length).
> `deepencoder.py:267-292` `forward`; `:274-283` uses `patch_embeds` arg; `:286-287` cat class token; `:290` add `get_abs_pos(...)`
> `deepencoder.py:243-265` `CLIPVisionEmbeddings.__init__` (class_embedding, patch_embedding Conv2d, position_embedding Embedding(num_positions=num_patches+1))

**[SPEC-049] CLIP forward = embeddings -> pre_layrnorm -> 24-layer transformer.** Each
`NoTPTransformerBlock`: `h = x + attn(layer_norm1(x))`; `out = h + mlp(layer_norm2(h))`. MLP uses
`quick_gelu(x) = x*sigmoid(1.702*x)`, `fc2(quick_gelu(fc1(x)))`. Attention is full SDPA (no causal
mask), qkv_proj bias=True, out_proj bias=True.
> `deepencoder.py:498-511` `VitModel.forward` = embeddings -> pre_layrnorm -> transformer
> `deepencoder.py:392-396` block residual order
> `deepencoder.py:295-309` `NoTPFeedForward` (`fc2(quick_gelu(fc1(x)))`)
> `deepencoder.py:237-239` `quick_gelu`
> `deepencoder.py:314-371` `NoTPAttention` (SDPA, `attn_mask=None`)
> `deepencoder.py:470-473` `pre_layrnorm = LayerNorm(hidden_size, eps=1e-5)`

**[SPEC-050] CLIP call signature in tower.** `vision_model(image, sam_features)` — the SAM output is
the second positional arg (`patch_embeds`).
> `modeling_unlimitedocr.py:501` `local_features_2 = vision_model(patches, local_features_1)`
> `modeling_unlimitedocr.py:508` `global_features_2 = vision_model(image_ori, global_features_1)`

### 4.3 Feature concat + projector

**[SPEC-051] Hybrid feature = concat(CLIP[:,1:], SAM_flat) along channels.** Drop CLIP class token
(`[:,1:]`); flatten SAM `x3` `flatten(2).permute(0,2,1)` (=> `B, HW, 1024`); concat to
`B, HW, 1024+1024=2048`. Then `projector` (linear 2048->1280).
> `modeling_unlimitedocr.py:503` `local_features = torch.cat((local_features_2[:,1:], local_features_1.flatten(2).permute(0,2,1)), dim=-1)`
> `modeling_unlimitedocr.py:504` `local_features = self.projector(local_features)`
> `modeling_unlimitedocr.py:509-510` same for global
> `deepencoder.py:31-32, 110, 169` `MlpProjector` linear path: `nn.Linear(input_dim, n_embed)`; `forward` returns `self.layers(x)`

**[SPEC-052] Projector = single `nn.Linear`.** projector_type "linear" => `self.layers = nn.Linear(2048,1280)`.
> `deepencoder.py:31-32` `elif cfg.projector_type == "linear": modules = nn.Linear(cfg.input_dim, cfg.n_embed)`

---

## 5. CONNECTOR (image_newline / view_seperator + masked_scatter)

In `UnlimitedOCRModel.forward` (`modeling_unlimitedocr.py:449-592`).

**[SPEC-060] Learned connector params.** `image_newline` and `view_seperator` are
`nn.Parameter(randn(1280) * (1/sqrt(1280)))`.
> `modeling_unlimitedocr.py:442-444` `embed_std = 1/sqrt(1280); image_newline = nn.Parameter(randn(n_embed)*embed_std); view_seperator = nn.Parameter(randn(n_embed)*embed_std)`

**[SPEC-061] Vision branch trigger condition.** Vision processing runs only when `sam_model` present,
`images` present, `(input_ids.shape[1] != 1 or training)` (i.e. prefill, not decode), AND
`sum(images[0][1]) != 0` (the `image_ori` tensor is non-zero).
> `modeling_unlimitedocr.py:480` exact guard
> `image[0]` = patches/crop, `image[1]` = image_ori (`:489-490`)

**[SPEC-062] CROP branch (`sum(patches) != 0`).** Build local + global hybrid features (SPEC-051),
then spatially arrange:
- `h = w = sqrt(global_hw) = 16`; `h2 = w2 = sqrt(local_hw) = 10`.
- Global: view to `(h,w,1280)`, append `image_newline` as an extra column =>
  `cat([global, image_newline.expand(h,1,n_dim)], dim=1)` -> view `(-1,1280)`. So each of 16 rows is
  16 patch embeds + 1 newline embed = 17; total 16*17 = 272 global tokens.
- Local: view `(height_crop_num, width_crop_num, h2, w2, n_dim).permute(0,2,1,3,4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim)`; append `image_newline` column -> view `(-1,1280)`.
- Final per-image: `cat([local_features, global_features, view_seperator[None,:]], dim=0)`.
> `modeling_unlimitedocr.py:496-541` full crop branch
> `:517-521` h/w and h2/w2; `:525-531` global view + newline + flatten
> `:534-538` local permute/reshape + newline
> `:540` `global_local_features = torch.cat([local_features, global_features, self.view_seperator[None,:]], dim=0)`

**[SPEC-063] NO-CROP branch (`sum(patches)==0`, e.g. infer_multi / single global).** For each image in
`image_ori` (num_imgs): build global hybrid features, view `(h,w,1280)`, append `image_newline`
column, flatten, then `cat([global_features, view_seperator[None,:]])`. Appended per image.
> `modeling_unlimitedocr.py:551-573` no-crop branch
> `:572` `global_local_features = torch.cat([global_features, self.view_seperator[None,:]], dim=0)`

**[SPEC-064] masked_scatter into text embeddings.** Concatenate all per-image feature blocks for the
batch element, then `inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), features)`.
The mask True positions (count must equal feature rows) are overwritten with vision features in order.
> `modeling_unlimitedocr.py:578-584`
> `images_in_this_batch = torch.cat(images_in_this_batch, dim=0)`
> `inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)`

**[SPEC-065] inputs_embeds source.** If not provided, `inputs_embeds = get_input_embeddings()(input_ids)`
(the decoder `embed_tokens`). After scatter, `input_ids=None` is passed to the parent decoder forward.
> `modeling_unlimitedocr.py:468-470` embed; `:587-592` parent forward with `input_ids=None, inputs_embeds=inputs_embeds`

**[SPEC-066 — ORDERING INVARIANT]** The `images_seq_mask` token layout (SPEC-028) must produce exactly:
(local block tokens) then (global block tokens) then (1 view-separator) per crop image — matching the
feature concat order `[local, global, view_seperator]` (SPEC-062). For no-crop: (global) then (1
separator). The Rust port MUST replicate this exact interleave so masked_scatter aligns. The newline
embed lands at the per-row trailing token; the separator at the per-image trailing token.
> Token side: `modeling_unlimitedocr.py:913-918`; feature side: `:534-541`.

---

## 6. DECODER (DeepseekV2 / Llama-MHA hybrid)

In `modeling_deepseekv2.py`.

**[SPEC-070] Decoder stack.** `DeepseekV2Model`: `embed_tokens` (Embedding vocab x 1280, padding_idx =
pad_token_id), `num_hidden_layers=12` `DeepseekV2DecoderLayer`s, final `DeepseekV2RMSNorm`.
> `modeling_deepseekv2.py:1610-1631` `__init__`
> `modeling_deepseekv2.py:1615-1617` embed_tokens; `:1618-1623` layers; `:1626` norm

**[SPEC-071] RMSNorm.** float32 variance, `x * rsqrt(mean(x^2)+eps)`, then `weight * x.to(dtype)`,
`eps=rms_norm_eps=1e-6`.
> `modeling_deepseekv2.py:96-110` `DeepseekV2RMSNorm.forward`

**[SPEC-072] Decoder layer forward (pre-norm residual).**
`h = x + self_attn(input_layernorm(x))`; `out = h + mlp(post_attention_layernorm(h))`.
> `modeling_deepseekv2.py:1453-1473`

**[SPEC-073] Attention class selection.** `use_mla=False` => `attn_implementation = "mha_" + _attn_implementation`.
The map: `"mha_eager" -> SlidingWindowLlamaAttention`. (No `mha_flash_attention_2` entry.) So the
runtime decoder attention is **SlidingWindowLlamaAttention** (a LlamaAttention subclass), NOT the
DeepseekV2 MLA attention.
> `modeling_deepseekv2.py:1398-1405` `if config.use_mla: "mla_"+... else "mha_"+...`
> `modeling_deepseekv2.py:1380-1389` `ATTENTION_CLASSES`: `"mha_eager": SlidingWindowLlamaAttention`

**[SPEC-074] dense-vs-MoE per layer.** Layer `i` is `DeepseekV2MoE` iff
`n_routed_experts is not None AND i >= first_k_dense_replace(=1) AND i % moe_layer_freq(=1) == 0`.
=> layer 0 is `DeepseekV2MLP` (dense); layers 1..11 are MoE.
> `modeling_deepseekv2.py:1407-1415`

**[SPEC-075] Dense MLP (layer 0 & shared experts).** SwiGLU: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`,
`act_fn = silu`. Dense intermediate = `intermediate_size=6848`.
> `modeling_deepseekv2.py:383-399` `DeepseekV2MLP`

**[SPEC-076] MoE forward (inference path `moe_infer`).**
- Gate produces `topk_idx, topk_weight` (SPEC-077).
- Shared experts: `y = moe_infer(...) + shared_experts(identity)` where shared expert is a
  `DeepseekV2MLP` with `intermediate_size = moe_intermediate_size(896) * n_shared_experts(2) = 1792`.
- `moe_infer`: route each token to its top-6 experts, run each expert
  (`DeepseekV2MLP(intermediate_size=896)`), weight by `topk_weight`, sum.
> `modeling_deepseekv2.py:608-628` `forward` (eval -> `moe_infer`, `:625`); `:626-627` shared add
> `modeling_deepseekv2.py:602-606` shared expert sizing
> `modeling_deepseekv2.py:630-704` `moe_infer` (sort by expert, per-expert matmul, scatter back, weight & sum)

**[SPEC-077] MoEGate (greedy softmax top-k).**
- `logits = F.linear(hidden.float(), weight.float())`; `scores = softmax(logits, dim=-1, float32)`
  (scoring_func='softmax').
- topk_method='greedy': `topk_weight, topk_idx = torch.topk(scores, k=6, sorted=False)`.
- norm: since `top_k>1 and norm_topk_prob=False` => `topk_weight = topk_weight * routed_scaling_factor(1.0)`
  (NO renormalization).
> `modeling_deepseekv2.py:433-447` score; `:450-453` greedy topk; `:502-507` norm branch (False path `:506-507`)
> Gate weight `nn.Parameter((n_routed_experts, hidden_size)) = (64,1280)` (`:419-421`).

**[SPEC-078] RoPE (Llama-style, applied inside SlidingWindowLlamaAttention).** Uses
`LlamaRotaryEmbedding(config)` (created if absent). theta=`rope_theta=10000`, head_dim=128,
`rope_scaling=None`. NOTE: the decoder runtime uses the **Llama** `apply_rotary_pos_emb`
(`_llama_apply_rotary_pos_emb`), NOT the DeepseekV2 interleaved variant — the DeepseekV2 `apply_rotary_pos_emb`
(with the `view(...,d//2,2).transpose` GPT-NeoX interleave at `:370-377`) is only used by the unused
MLA attention classes.
> `modeling_deepseekv2.py:1238-1240` create `LlamaRotaryEmbedding(config=config)`
> `modeling_deepseekv2.py:1290-1291` `cos,sin = self.rotary_emb(value_states, position_ids); q,k = _llama_apply_rotary_pos_emb(q,k,cos,sin)`
> `modeling_deepseekv2.py:37-42` imports `apply_rotary_pos_emb as _llama_apply_rotary_pos_emb, repeat_kv as _llama_repeat_kv`

**[SPEC-079] Position IDs.** Built in `prepare_inputs_for_generation` /  model forward:
`position_ids = arange(past_len, seq_len+past_len)`; or from attention_mask cumsum when present.
> `modeling_deepseekv2.py:1694-1702` model forward arange
> `modeling_unlimitedocr.py:732-738` generation position_ids

**[SPEC-080] 4D causal mask handling.** Decode (`seq_length==1 and past>0`) => `attention_mask=None`
(no mask). Prefill => `_prepare_4d_causal_attention_mask(...)`.
> `modeling_deepseekv2.py:1707-1724`

**[SPEC-081] LM head + logits.** `logits = lm_head(hidden_states).float()`. lm_head is a separate
non-tied Linear (`tie_word_embeddings=False`).
> `modeling_unlimitedocr.py:662-664` `hidden_states=outputs[0]; logits=self.lm_head(hidden_states); logits=logits.float()`
> `modeling_unlimitedocr.py:606` lm_head def; `config.json:100` `"lm_head": true`

---

## 7. SlidingWindowLlamaAttention — RING BUFFER SEMANTICS

`modeling_deepseekv2.py:1232-1377`. This is the load-bearing custom kernel. Window `W = config._ring_window = 128`.

**[SPEC-090] Heads.** `num_heads=10`, `num_kv_heads=10` (so `num_key_value_groups=1`, `repeat_kv` is a
no-op), `head_dim=128`. Standard QKV: `q_proj/k_proj/v_proj/o_proj`, all `bias=attention_bias=False`.
Attention scale `1/sqrt(head_dim)=1/sqrt(128)`.
> `modeling_deepseekv2.py:1276-1281` dims; `:1286-1288` projections; `:1299` scale; `:1370-1371` repeat_kv

**[SPEC-091] Three regimes** (decided per layer, per forward):
1. **True prefill** — `W is None OR past_kv is None OR (q_len>1 and layer not yet in `_prefill_length`)`:
   standard full causal attention via `_attn_forward()` (uses `past_kv.update` to append to cache).
   On the first prefill with `q_len>1`, record `_prefill_length[layer_idx] = current cache length`.
   > `:1310-1323`
2. **Warmup decode** — `cur_len < prefill_len + W`: standard `_attn_forward()` (cat-append to cache)
   until the ring region (W slots after prefill) is full; when full, set `_ring_pos[layer_idx]=0`.
   > `:1325-1342`
3. **Steady-state ring** — `cur_len >= prefill_len + W`: do NOT grow the cache. Compute new K,V
   (with RoPE), overwrite ring slots IN PLACE:
   for each of `q_len` new tokens, `slot = prefill_len + ring_pos`,
   `kcache[:,:,slot:slot+1,:] = new_k; vcache[...] = new_v; ring_pos = (ring_pos+1) % W`.
   Then attention over the FULL cache (prefill + W ring slots) with NO causal mask (decode q_len=1).
   > `:1344-1377`

**[SPEC-092] Cache compatibility shims.** `_get_kcache/_get_vcache` handle both old
(`cache.key_cache[i]`) and new (`cache.layers[i].keys`) DynamicCache layouts.
> `modeling_deepseekv2.py:1248-1257`

**[SPEC-093] State stored ON the cache object.** `past_kv._prefill_length` (dict layer->len),
`past_kv._ring_pos` (dict layer->pos). These persist across decode steps for the generation.
> `:1319-1322, 1326-1331, 1339-1347, 1350, 1363-1367`

**[SPEC-094] Effective attention window.** Decode attends over `prefill_len + W` keys: ALL prefill
tokens (image+prompt, never evicted) plus the last `W=128` generated tokens (ring). This is "R-SWA"
(retain-prefix sliding window attention): prefix is fully retained, only the generated tail slides.
> Derived from `:1335` (`cur_len < prefill_len + W` warmup) and `:1362-1371` (ring over `prefill_len`..`prefill_len+W`).

**[SPEC-095 — PORT INVARIANT]** RoPE position for a ring-overwritten token uses the TRUE absolute
`position_ids` (monotonic, from generation), not the ring slot index. The KV is stored at a recycled
physical slot but carries its original RoPE phase. The Rust port must decouple physical slot from
logical position.
> `:1358-1359` RoPE applied with real `position_ids` before slot overwrite at `:1362-1366`.

**[SPEC-096] prepare_inputs_for_generation ring awareness.** When `past_kv` has `_prefill_length` and
`past_length>0`, only the last token is fed (`input_ids = input_ids[:, -1:]`); images passed only on
prefill (`_is_prefill` gate).
> `modeling_unlimitedocr.py:708-711` ring decode single-token; `:760-772` images only on prefill

---

## 8. SAMPLER (generation)

**[SPEC-100] Greedy by default.** `do_sample = temperature > 0`; default `temperature=0.0` =>
greedy (argmax). When `>0`, `temperature` passed through; sampling otherwise standard HF `.generate`.
> `modeling_unlimitedocr.py:1007-1008` `do_sample=temperature>0, temperature=temperature if temperature>0 else None`
> `modeling_unlimitedocr.py:1020` `output_ids = self.generate(**gen_kwargs)`

**[SPEC-101] EOS / max_length.** `eos_token_id = tokenizer.eos_token_id` (=1), `max_length=32768`,
`use_cache=True`.
> `modeling_unlimitedocr.py:1009-1012`

**[SPEC-102] no_repeat_ngram options.** If `no_repeat_ngram_size>0 AND ngram_window>0`: use custom
`SlidingWindowNoRepeatNgramProcessor`. Elif `no_repeat_ngram_size>0`: use HF builtin
`no_repeat_ngram_size`. Else: none.
> `modeling_unlimitedocr.py:1014-1017`

**[SPEC-103] SlidingWindowNoRepeatNgramProcessor.** For each batch row's token sequence: if
`len < ngram_size` skip. `search_start = max(0, len - window)`, `search_end = len - ngram_size + 1`.
For `ngram_size>1`, `current_prefix = last (ngram_size-1) tokens`. Scan `[search_start, search_end)`:
if ngram's prefix matches `current_prefix` (or ngram_size==1), ban its last token. Remove whitelist
tokens; set `scores[batch, banned] = -inf`.
> `modeling_unlimitedocr.py:354-383` full class
> Aligned with SGLang `DeepseekOCRNoRepeatNGramLogitProcessor` (`:355-356`).

---

## 9. POSTPROCESS (ref/det regex, bbox/999, markdown, <PAGE>)

**[SPEC-110] Decode + strip EOS.** Decode `output_ids[0, prompt_len:]`; if ends with stop string
`'<｜end▁of▁sentence｜>'` strip it; `.strip()`.
> `modeling_unlimitedocr.py:1049-1054`, `:1070-1078`, `:1259-1263`

**[SPEC-111] re_match — ref/det extraction.** Two regexes:
- `ref_pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'` (DOTALL): captures (full, label, box).
- `det_pattern = r'(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)'`: standalone det.
Classify each match: label `=='image'` (or full contains `<|ref|>image<|/ref|>`) -> `mathes_image`,
else `mathes_other`. Returns `(matches, mathes_image, mathes_other)`.
> `modeling_unlimitedocr.py:44-59`

**[SPEC-112] Coordinate parsing.** `extract_coordinates_and_label`: `label_type = ref_text[1]`;
`cor_list = eval(ref_text[2])`; if first element is scalar, wrap as `[cor_list]` (single box).
> `modeling_unlimitedocr.py:62-73`

**[SPEC-113] bbox scaling — divide by 999.** Each box `(x1,y1,x2,y2)` (normalized 0..999) is mapped to
pixels: `x = int(coord / 999 * image_width)` etc.
> `modeling_unlimitedocr.py:104-111` `x1 = int(x1 / 999 * image_width)` (and y1/x2/y2)

**[SPEC-114] image-label crops.** For `label_type=='image'`: crop the region and save
`{output}/images/{prefix}{idx}.jpg`; in markdown, replace the matched ref string with
`![](images/{idx}.jpg)\n`.
> `modeling_unlimitedocr.py:113-120` crop+save; `:1085-1086` markdown image replace

**[SPEC-115] other-label cleanup.** Replace each `mathes_other` match with `''`; also globally
`\coloneqq -> :=` and `\eqqcolon -> =:`. Write `result.md`.
> `modeling_unlimitedocr.py:1088-1095`

**[SPEC-116] Bounding-box overlay drawing.** `draw_bounding_boxes`: random color per ref; `title`
labels drawn with width 4 + filled overlay, others width 2; label text drawn above box. Saved as
`result_with_boxes.jpg`.
> `modeling_unlimitedocr.py:76-145`, `:1136`

**[SPEC-117] Geometry/line special case.** If output contains `'line_type'`, `eval(outputs)` parsed as
a dict with `['Line']['line']`, `['line_type']`, `['line_endpoint']` and plotted to `geo.jpg`.
> `modeling_unlimitedocr.py:1097-1134`

**[SPEC-118] Multi-page output (`<PAGE>`).** `infer_multi`: split decoded output on `'<PAGE>'`
(`outputs.split('<PAGE>')[1:]`), process each page independently (ref/det per page,
`page_{i}_` prefix), rejoin as `'<PAGE>\n' + '\n<PAGE>\n'.join(processed_pages)`.
> `modeling_unlimitedocr.py:1269-1297`

**[SPEC-119] test_compress metric.** `compression_ratio = output_text_tokens / valid_img_tokens`.
> `modeling_unlimitedocr.py:1058-1065`

---

## 10. UNRESOLVED / BLOCKERS

**[UNRESOLVED-1] BPE merges / tokenizer model.** `tokenizer_config.json` (snapshot) gives the
special-token table and class (`LlamaTokenizerFast`, `legacy=true`), but the actual BPE vocab/merges
(`tokenizer.json`) is NOT in the snapshot set. Exact text->id encoding for arbitrary prompt/output
strings cannot be reproduced from these files alone. **Needs:** `tokenizer.json`. (Tracks the
tokenizer-parity OQ.)

**[UNRESOLVED-2] NVFP4 / weight quantization layout.** Source `torch_dtype="bfloat16"`
(`config.json:62`) and `model.safetensors.index.json` references a single bf16 shard. Any NVFP4 path
is external to these snapshots. **Needs:** the external NVFP4 repo / quant spec if the Rust port
targets NVFP4. Not resolvable here.

**[UNRESOLVED-3] CLIP `get_abs_pos` interpolation for the fused tower.** The CLIP embeddings receive
SAM features as `patch_embeds` (SPEC-048); `get_abs_pos` interpolates the 224/14=16x16 CLIP positional
grid to the SAM-derived sequence length (16x16=256 for 1024 input => `src_size==tgt_size==16`, so no
interpolation in the 1024 case — `deepencoder.py:218,234-235` returns `abs_pos` unchanged). For other
sizes bicubic interpolation applies. This resolves for the pinned 1024 base size but the port must
implement the bicubic branch for completeness. Cite `deepencoder.py:199-235`.

**[UNRESOLVED-4] `time` import / minor dead code.** `import time` at module top + commented timing
blocks are no-ops; not load-bearing. No spec impact.
