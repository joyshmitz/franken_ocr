#!/usr/bin/env python3
"""Generate the baidu/Unlimited-OCR REFERENCE OCR text for franken_ocr parity.

Runs the *pinned* baidu reference implementation (truth-pack modeling code, the
3a7f4dbb revision) on CPU and captures the decoded markdown per page. This is the
oracle franken_ocr's native forward must match.

The upstream `infer()` is hardcoded for CUDA + bf16 autocast. We port it to CPU
*without editing the modeling code* by monkeypatching:
  - torch.Tensor.cuda  -> identity (keep tensors on CPU)
  - torch.autocast("cuda", ...) -> nullcontext (run in the loaded dtype, fp32)
  - torch.cuda.is_available -> False (defensive; flash-attn stays disabled)

We load in bfloat16 — the model ships bf16, `infer()` hardcodes bf16 image
tensors, and matching the dtype keeps the whole CPU graph bf16 (faithful to the
upstream bf16+CUDA numerics, and ~6.7GB resident vs ~13GB for fp32). Decoding is
greedy (temperature=0) so the reference is deterministic. The native engine is
int8/int4-quantized, so the parity bar is the decoded TEXT (CER / exact-match),
not bit-identical logits; bf16-CPU vs bf16-CUDA drift does not change greedy OCR
output materially.

Usage:
  run_baidu_reference.py --model DIR --pages-dir DIR --out DIR \
      [--only page_0009.png] [--max-length 8192] [--threads N]
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def install_cpu_patches():
    import torch

    # 1) tensor.cuda() -> identity (no device move)
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[attr-defined]

    # 2) torch.autocast("cuda", ...) -> nullcontext (stay in loaded dtype)
    _orig_autocast = torch.autocast

    class _AutocastShim(contextlib.AbstractContextManager):
        def __init__(self, device_type, *a, **k):
            if device_type == "cuda":
                self._cm = contextlib.nullcontext()
            else:
                self._cm = _orig_autocast(device_type, *a, **k)

        def __enter__(self):
            return self._cm.__enter__()

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    torch.autocast = _AutocastShim  # type: ignore[assignment]

    # 3) defensive: no CUDA visible
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pages-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="single page filename to run")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--crop-mode", action="store_true", default=False)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=35)
    ap.add_argument("--ngram-window", type=int, default=1024)
    ap.add_argument("--prompt", default="<image>document parsing.")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    install_cpu_patches()
    import torch
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    from transformers import AutoModel, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[ref] loading tokenizer from {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[ref] loading model (bf16, CPU) — this reads the 6.67GB shard ...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    print(f"[ref] model loaded in {time.time()-t0:.1f}s; threads={torch.get_num_threads()}", flush=True)

    pages_dir = Path(args.pages_dir)
    if args.only:
        pages = [pages_dir / args.only]
    else:
        pages = sorted(pages_dir.glob("page_*.png"))
    print(f"[ref] {len(pages)} page(s) to process", flush=True)

    index = []
    for p in pages:
        print(f"[ref] === {p.name} ===", flush=True)
        t0 = time.time()
        with torch.no_grad():
            text = model.infer(
                tok,
                prompt=args.prompt,
                image_file=str(p),
                output_path=str(out / "_scratch"),
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=args.crop_mode,
                eval_mode=True,
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window,
                temperature=0.0,
            )
        dt = time.time() - t0
        text = text if isinstance(text, str) else str(text)
        md_path = out / f"{p.stem}.md"
        md_path.write_text(text)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ntoks = len(tok(text, add_special_tokens=False)["input_ids"])
        rec = {"page": p.name, "chars": len(text), "tokens": ntoks,
               "sha256": sha, "seconds": round(dt, 2), "md": md_path.name}
        index.append(rec)
        print(f"[ref] {p.name}: {len(text)} chars, {ntoks} toks, {dt:.1f}s, sha {sha[:12]}", flush=True)
        (out / "INDEX.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")

    print(f"[ref] done; wrote {len(index)} reference(s) to {out}", flush=True)


if __name__ == "__main__":
    main()
