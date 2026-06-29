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
