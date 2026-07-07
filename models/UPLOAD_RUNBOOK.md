# Model weights — build + upload runbook

`focr` downloads the weights on demand from the mirrors in
[`manifest.json`](./manifest.json) (see `focr pull` / the first-run prompt). The
weights themselves are **never committed** (`*.focrq` is gitignored); only the
small manifest lives in the repo. This runbook reproduces the artifacts and
uploads them so the manifest URLs resolve.

## What the manifest points at

| artifact | size | sha256 |
|---|---|---|
| `unlimited-ocr.int8.focrq` (reassembled) | 3 914 093 440 | `d8c5fcf223c8e062af63f6b86964d099e2c5a5b272ae096a09433aaf5510a440` |
| `unlimited-ocr.int8.focrq.part00` | 1 957 046 720 | `e58503fb1700a56cff71d2c136c223efa5c10c38604901987904467267c815e3` |
| `unlimited-ocr.int8.focrq.part01` | 1 957 046 720 | `6a647a04fece7bced666c0e56581969c1a799e8447c64e9abf37ee10c322ec7d` |
| `tokenizer.json` | 9 979 544 | `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4` |

The two `.focrq` parts are a plain byte split (concatenation = the file); GitHub
caps release assets at 2 GB, so the 3.9 GB blob ships as 2 parts that
`focr pull` reassembles + sha256-verifies. The `tokenizer.json` is already
public at `baidu/Unlimited-OCR` (the manifest lists that as its primary mirror),
so it needs no upload — the GitHub mirror is an optional fallback.

## 1. Build the artifacts (reproducible)

```bash
# From the bf16 safetensors (HF baidu/Unlimited-OCR), produce the int8 .focrq:
focr convert /path/to/model-00001-of-000001.safetensors \
  -o unlimited-ocr.int8.focrq --quant int8
# -> 3 914 093 440 bytes, sha256 d8c5fcf2…  (deterministic; verify it matches)

# Split for GitHub's 2 GB asset cap (2 equal parts of 1 957 046 720 bytes):
split -b 1957046720 -d unlimited-ocr.int8.focrq unlimited-ocr.int8.focrq.part

# Confirm the part sha256s match the table above:
shasum -a 256 unlimited-ocr.int8.focrq.part00 unlimited-ocr.int8.focrq.part01
```

## 2. Upload to GitHub Releases (mirror 1)

```bash
gh release create models-v1 \
  --repo Dicklesworthstone/franken_ocr \
  --title "Model weights (int8) — unlimited-ocr" \
  --notes "int8 .focrq weights for franken_ocr; see models/manifest.json"

gh release upload models-v1 \
  --repo Dicklesworthstone/franken_ocr \
  unlimited-ocr.int8.focrq.part00 \
  unlimited-ocr.int8.focrq.part01
# (tokenizer.json optional here; the manifest's primary tokenizer mirror is baidu HF)
```

## 3. Upload to Hugging Face (mirror 2)

Create a weights repo (the manifest expects `Dicklesworthstone/franken_ocr-weights`;
change both places if you use another name), then:

```bash
huggingface-cli upload Dicklesworthstone/franken_ocr-weights \
  unlimited-ocr.int8.focrq.part00 unlimited-ocr.int8.focrq.part00
huggingface-cli upload Dicklesworthstone/franken_ocr-weights \
  unlimited-ocr.int8.focrq.part01 unlimited-ocr.int8.focrq.part01
```

## 4. Verify end-to-end

```bash
focr pull            # uses models/manifest.json by default; downloads + verifies
focr ocr page.png    # resolves the cached model, fully offline
```

`focr pull` tries each mirror in order, verifies every part's sha256 AND the
reassembled file's sha256, and installs into `~/.cache/franken_ocr/models/`. If
a sha mismatches it errors loudly rather than caching corrupt bytes.

> Updating the weights later: re-run `focr convert`, re-split, bump the release
> tag (e.g. `models-v2`) + the URLs/sha256s in `manifest.json`, and commit the
> manifest. Old binaries keep working against the old tag until they pull again.

---

# Zoo models — smolvlm2 / onechart / tromr (bd-av64.7/.8)

Every asset is **under GitHub's 2 GB cap, so NO split parts** (the asymmetry vs
unlimited-ocr above). One GH release tag per model; the HF mirror uses a
**per-model subdirectory** (sidecar filenames are not unique across models —
smolvlm2 ships its own `tokenizer.json`). `focr pull <model>` installs each
non-primary model into its own cache subdir (`~/.cache/franken_ocr/models/<id>/`)
for the same reason.

Source of truth for every byte: the staged files on the USB zoo
(`/Volumes/USBNVME16TB/temp_agent_space/zoo/<model>/`), hashed 2026-07-06; the
committed `manifest.json` carries the same sizes + sha256s (the embedded-manifest
lint test `builtin_manifest_publishes_the_zoo_and_lints_clean` cross-checks the
shape).

## What the manifest points at

| release tag | asset | size | sha256 (first 16) |
|---|---|---|---|
| `models-smolvlm2-v1` | `smolvlm2.int8.focrq` | 1 087 397 293 | `4ad2ac89e47c83ad…` |
| `models-smolvlm2-v1` | `tokenizer.json` | 3 548 256 | `5ece781dc8d2b2f3…` |
| `models-onechart-v1` | `onechart.int8.focrq` | 362 863 824 | `618189a8e975f0cf…` |
| `models-onechart-v1` | `vocab.json` | 999 355 | `32b29acf82d33334…` |
| `models-onechart-v1` | `merges.txt` | 456 318 | `1ce1664773c50f3e…` |
| `models-onechart-v1` | `added_tokens.json` | 82 | `e1b04af1435ff5b4…` |
| `models-tromr-v1` | `tromr.int8.focrq` | 61 107 485 | `cced11c0f05656dd…` |
| `models-tromr-v1` | `tromr.focrq` (f32) | 86 168 002 | `a9d41485a98534ad…` |
| `models-tromr-v1` | `tokenizer_rhythm.json` | 10 743 | `603bfef760e8424f…` |
| `models-tromr-v1` | `tokenizer_pitch.json` | 2 682 | `2382e8b20c147329…` |
| `models-tromr-v1` | `tokenizer_lift.json` | 979 | `b61ba09cecd5bc34…` |
| `models-tromr-v1` | `tokenizer_note.json` | 830 | `504d886d11e3c1fe…` |

Runtime-required file sets (verified against the loaders 2026-07-06 — a pull
missing any of these is a broken pull): smolvlm2 = focrq + `tokenizer.json`;
onechart = focrq + `vocab.json` + `merges.txt` + `added_tokens.json`
(`Tokenizer::from_opt_dir`); tromr = focrq + all four `tokenizer_*.json`
(`MusicTokenizer::from_dir`). TrOMR's `config.yaml` is **convert-time only**
(zero runtime references) — attach it to the GH release for provenance if you
like, but it is deliberately NOT in the manifest. TrOMR publishes **both
quants** since bd-av64.12 ran the lossless proof (40 decoder GEMMs int8,
committed golden byte-identical, corpus gate delta 0 — divergence ledgered as
DISC-005): the default `focr pull tromr` now fetches `tromr.int8.focrq`;
`--quant f32` fetches the bit-exact reference artifact.

## Upload — GitHub Releases (mirror 1, copy-paste)

```bash
Z=/Volumes/USBNVME16TB/temp_agent_space/zoo

gh release create models-smolvlm2-v1 --repo Dicklesworthstone/franken_ocr \
  --title "SmolVLM2-500M int8 weights (models-smolvlm2-v1)" \
  --notes "SmolVLM2-500M-Video-Instruct (HuggingFaceTB, Apache-2.0) int8 .focrq + tokenizer; see models/manifest.json"
gh release upload models-smolvlm2-v1 --repo Dicklesworthstone/franken_ocr \
  "$Z/smolvlm2/smolvlm2.int8.focrq" "$Z/smolvlm2/tokenizer.json"

gh release create models-onechart-v1 --repo Dicklesworthstone/franken_ocr \
  --title "OneChart int8 weights (models-onechart-v1)" \
  --notes "OneChart (kppkkp, Apache-2.0) int8 .focrq + OPT tokenizer triple; see models/manifest.json"
gh release upload models-onechart-v1 --repo Dicklesworthstone/franken_ocr \
  "$Z/onechart/onechart.int8.focrq" "$Z/onechart/vocab.json" \
  "$Z/onechart/merges.txt" "$Z/onechart/added_tokens.json"

gh release create models-tromr-v1 --repo Dicklesworthstone/franken_ocr \
  --title "Polyphonic-TrOMR f32 weights (models-tromr-v1)" \
  --notes "Polyphonic-TrOMR (NetEase, Apache-2.0) f32 .focrq + the four music tokenizer tables; see models/manifest.json"
gh release upload models-tromr-v1 --repo Dicklesworthstone/franken_ocr \
  "$Z/tromr/tromr.focrq" "$Z/tromr/tokenizer_rhythm.json" \
  "$Z/tromr/tokenizer_pitch.json" "$Z/tromr/tokenizer_lift.json" \
  "$Z/tromr/tokenizer_note.json"
```

## Upload — Hugging Face mirror (per-model SUBDIRECTORIES)

```bash
Z=/Volumes/USBNVME16TB/temp_agent_space/zoo
R=Dicklesworthstone/franken_ocr-weights

huggingface-cli upload "$R" "$Z/smolvlm2/smolvlm2.int8.focrq" smolvlm2/smolvlm2.int8.focrq
huggingface-cli upload "$R" "$Z/smolvlm2/tokenizer.json"      smolvlm2/tokenizer.json
huggingface-cli upload "$R" "$Z/onechart/onechart.int8.focrq" onechart/onechart.int8.focrq
huggingface-cli upload "$R" "$Z/onechart/vocab.json"          onechart/vocab.json
huggingface-cli upload "$R" "$Z/onechart/merges.txt"          onechart/merges.txt
huggingface-cli upload "$R" "$Z/onechart/added_tokens.json"   onechart/added_tokens.json
huggingface-cli upload "$R" "$Z/tromr/tromr.focrq"            tromr/tromr.focrq
huggingface-cli upload "$R" "$Z/tromr/tokenizer_rhythm.json"  tromr/tokenizer_rhythm.json
huggingface-cli upload "$R" "$Z/tromr/tokenizer_pitch.json"   tromr/tokenizer_pitch.json
huggingface-cli upload "$R" "$Z/tromr/tokenizer_lift.json"    tromr/tokenizer_lift.json
huggingface-cli upload "$R" "$Z/tromr/tokenizer_note.json"    tromr/tokenizer_note.json
```

## Post-upload spot-check (before the full bd-av64.9 verification)

```bash
# Every GH URL in the manifest must answer 200 with the manifest's size:
python3 - <<'PY'
import json, urllib.request
m = json.load(open('models/manifest.json'))
for mid, e in m['models'].items():
    files = [q['focrq'] for q in e['quants'].values()] + [e['tokenizer']] + e.get('sidecars', [])
    for f in files:
        url = f['parts'][0]['urls'][0]
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'))
        size = int(r.headers['Content-Length'])
        print(('OK ' if size == f['size'] else 'SIZE MISMATCH ') + f"{mid} {f['filename']} {size}")
PY
```

Then run the full bd-av64.9 verification: clean-cache `focr pull
smolvlm2|onechart|tromr` + one real inference per model.
