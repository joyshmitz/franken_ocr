#!/usr/bin/env python3
"""Dump baidu per-stage vision activations as .npy fixtures for franken_ocr parity.

Hooks the reference model's vision submodules and runs one page through infer()
(base mode), capturing each stage boundary so the native engine can be compared
stage-by-stage (cosine >= 0.9999):
  sam_in.npy        [1,3,1024,1024]  normalized image tensor SAM receives
  sam_out.npy       SAM ViT-B output (global_features_1, spatial)
  clip_out.npy      [1,257,1024]     CLIP-L output (before dropping CLS)
  projector_out.npy [1,256,1280]     hybrid concat -> projector (bridge output)

Feeding franken_ocr's vision_sam::forward the SAME sam_in.npy decouples
preprocessing parity from vision-tower parity.

Usage: dump_stage_activations.py --model DIR --page PNG --out DIR
"""
import argparse
import contextlib
from pathlib import Path

import numpy as np


def install_cpu_patches():
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[attr-defined]
    _orig = torch.autocast

    class _Shim(contextlib.AbstractContextManager):
        def __init__(self, device_type, *a, **k):
            self._cm = contextlib.nullcontext() if device_type == "cuda" else _orig(device_type, *a, **k)

        def __enter__(self):
            return self._cm.__enter__()

        def __exit__(self, *e):
            return self._cm.__exit__(*e)

    torch.autocast = _Shim  # type: ignore[assignment]
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]


def to_np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    if isinstance(x, (tuple, list)):
        return to_np(x[0])
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    install_cpu_patches()
    import torch
    torch.set_num_threads(args.threads)
    from transformers import AutoModel, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()

    inner = model.model  # UnlimitedOCRModel
    acts = {}

    def save_in(name):
        def fn(module, inp, out_):
            acts[name + "_in"] = to_np(inp[0])
            acts[name + "_out"] = to_np(out_)
        return fn

    def save_out(name):
        def fn(module, inp, out_):
            acts[name + "_out"] = to_np(out_)
        return fn

    h = []
    h.append(inner.sam_model.register_forward_hook(save_in("sam")))
    h.append(inner.vision_model.register_forward_hook(save_out("clip")))
    h.append(inner.projector.register_forward_hook(save_out("projector")))

    print(f"[dump] running {Path(args.page).name} (base mode) ...", flush=True)
    with torch.no_grad():
        text = model.infer(
            tok, prompt="<image>document parsing.", image_file=args.page,
            output_path=str(out / "_scratch"), base_size=1024, image_size=1024,
            crop_mode=False, eval_mode=True, max_length=64,
            no_repeat_ngram_size=35, ngram_window=1024, temperature=0.0,
        )
    for hh in h:
        hh.remove()

    for k, v in acts.items():
        np.save(out / f"{k}.npy", v)
        print(f"[dump] {k}: shape={v.shape} dtype={v.dtype} -> {k}.npy", flush=True)
    print(f"[dump] decoded prefix: {text[:80]!r}", flush=True)
    print(f"[dump] wrote {len(acts)} activation fixtures to {out}", flush=True)


if __name__ == "__main__":
    main()
