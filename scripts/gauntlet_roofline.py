#!/usr/bin/env python3
"""Roofline floors for the head-to-head gauntlet (bd-re8.17; plan §9.1).

For each pipeline stage, computes the **compute floor** (int8 GEMM MACs ÷ the
machine's peak int8 throughput, plus f32 terms ÷ peak f32) and the **memory
floor** (bytes streamed ÷ DRAM bandwidth) and takes the **max** — what a
perfect kernel would hit. `docs/PERF_LEDGER.md` records that floor next to
every measured ratio: `dist_above_floor = focr_ms / floor_ms`, and a stage
>~1.3× above its floor is a named attackable lever.

The model shapes are transcribed from the source of truth in `src/` (each
constant cites its file); token counts and view counts are READ FROM the real
measurement file produced by `scripts/gauntlet_focr.sh` — a token-dependent
floor without a measurement file is REFUSED, never assumed. Floors are
analytic lower bounds: terms that are hard to bound tightly (norms, softmax,
embeds, activations) are *excluded*, which can only LOWER the floor and
therefore never flatters `dist_above_floor`.

The machine profile is explicit, echoed into the output, and overridable —
a floor is only meaningful relative to the recorded profile.

Usage:
  gauntlet_roofline.py --arch unlimited-ocr --precision int8 \
      --stages-json artifacts/perf/bd-re8.17/focr/focr_stages.json \
      [--stage decode_per_token ...] [--profile m4|m4-pro|m4-max] \
      [--dram-gb-s X] [--peak-int8-gmacs X] [--peak-f32-gflops X] --out FILE
  gauntlet_roofline.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

SCHEMA = "focr-gauntlet-roofline/v1"

STAGES = ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end")

# ── model shapes (transcribed from src; the src constants are the truth) ────

UNLIMITED = {
    # src/native_engine/decoder.rs::config
    "hidden": 1280,
    "layers": 12,
    "heads": 10,
    "head_dim": 128,
    "vocab": 129_280,
    "dense_inter": 6848,  # layer 0 (FIRST_K_DENSE_REPLACE = 1)
    # src/native_engine/moe.rs::config — layers 1..11 are MoE
    "moe_layers": 11,
    "routed_experts": 64,
    "experts_per_tok": 6,
    "moe_inter": 896,
    "shared_inter": 1792,  # MOE_INTERMEDIATE_SIZE * N_SHARED_EXPERTS (fused)
    # src/native_engine/rswa.rs — decode attention reads a 128-token ring
    "kv_window": 128,
}

GOT = {
    # src/native_engine/decoder_qwen2.rs::DecoderConfig::got_ocr2 (dense, no MoE)
    "hidden": 1024,
    "layers": 24,
    "heads": 16,
    "head_dim": 64,
    "vocab": 151_860,
    "dense_inter": 2816,
    "kv_window": None,  # full causal: the window is the whole context
}

SMOLVLM2 = {
    # src/native_engine/decoder_qwen2.rs::DecoderConfig::smolvlm2 (dense, GQA 15/5)
    "hidden": 960,
    "layers": 32,
    "heads": 15,
    "head_dim": 64,
    "kv_dim": 5 * 64,  # num_key_value_heads 5 × head_dim
    "vocab": 49_280,
    "dense_inter": 2560,  # SwiGLU: gate/up/down = 3 GEMMs
    # UNTIED f32 lm_head (model_arch.rs: lm_head_stored_int8: false)
    "head_f32": True,
    "kv_window": None,  # full causal
}

ONECHART = {
    # src/native_engine/decoder_qwen2.rs::DecoderConfig::onechart (OPT-125M)
    "hidden": 768,
    "layers": 12,
    "heads": 12,
    "head_dim": 64,
    "kv_dim": 768,  # MHA
    "vocab": 50_269,
    "relu_inter": 3072,  # OPT ReLU MLP: fc1/fc2 = 2 GEMMs (no gate)
    # TIED head runs against the f32 embed matrix (lm_head omitted from .focrq)
    "head_f32": True,
    "kv_window": None,  # full causal
}

VISION_SIGLIP = {
    # src/native_engine/vision_siglip.rs — SigLIP-B/16 per 512² frame
    "dim": 768,
    "depth": 12,
    "mlp": 3072,
    "tokens": 1024,  # (512/16)² patches, full (non-windowed) attention
    # token_compress.rs — pixel_shuffle is a free gather; the connector GEMM is
    # modality_projection Linear(12288→960) over 64 shuffled tokens per frame.
    "connector_macs": 64 * 12_288 * 960,
    "params": 12 * (4 * 768 * 768 + 2 * 768 * 3072) + 768 * 3 * 16 * 16 + 12_288 * 960,
}

VISION_SAM = {
    # src/native_engine/vision_sam.rs — SAM-ViT-B at 1024px / patch 16
    "dim": 768,
    "depth": 12,
    "mlp": 3072,
    "tokens": 64 * 64,
    "window": 14 * 14,
    "global_blocks": 4,
    # Vary neck + conv compressor (768→256 1×1, 256 3×3, 256→512 s2, 512→1024 s2)
    "neck_convs_macs": (
        64 * 64 * 768 * 256
        + 64 * 64 * 9 * 256 * 256
        + 32 * 32 * 9 * 256 * 512
        + 16 * 16 * 9 * 512 * 1024
    ),
    "params": 12 * (4 * 768 * 768 + 2 * 768 * 3072)
    + 768 * 3 * 16 * 16
    + 768 * 256
    + 9 * 256 * 256
    + 9 * 256 * 512
    + 9 * 512 * 1024,
}

VISION_CLIP = {
    # src/native_engine/vision_clip.rs — CLIP-L/14 style NoTP tower ([SPEC-047])
    "dim": 1024,
    "depth": 24,
    "ffn": 4096,
    "tokens": 257,  # 256 patch embeds from SAM + CLS
}

BRIDGE = {"tokens_per_view": 256, "proj_in": 2048, "proj_out": 1280}  # vision_bridge.rs

# ── machine profiles (explicit; every field overridable) ────────────────────
#
# peak_int8_gmacs: NEON SDOT = 16 int8 MACs/instr × 4 SIMD pipes = 64
# MACs/cycle/P-core (SMMLA is half-rate on M4 — perf playbook). peak_f32_gflops:
# 128-bit FMA = 4 MACs = 8 FLOPs/instr × 4 pipes = 32 FLOPs/cycle/P-core.
# E-cores are excluded (conservative capacity), which RAISES floor_ms slightly;
# the profile is recorded in the output so a row's floor is reproducible.
PROFILES = {
    "m4": {"dram_gb_s": 120.0, "p_cores": 4, "ghz": 4.4},
    "m4-pro": {"dram_gb_s": 273.0, "p_cores": 10, "ghz": 4.4},
    "m4-max": {"dram_gb_s": 546.0, "p_cores": 12, "ghz": 4.4},
}


def resolve_profile(args: argparse.Namespace) -> dict:
    base = PROFILES[args.profile]
    p_cores, ghz = base["p_cores"], base["ghz"]
    profile = {
        "name": args.profile,
        "dram_gb_s": args.dram_gb_s if args.dram_gb_s is not None else base["dram_gb_s"],
        "peak_int8_gmacs": (
            args.peak_int8_gmacs if args.peak_int8_gmacs is not None else p_cores * ghz * 64.0
        ),
        "peak_f32_gflops": (
            args.peak_f32_gflops if args.peak_f32_gflops is not None else p_cores * ghz * 32.0
        ),
        "note": "P-cores only, SDOT 64 int8-MAC/cycle/core, FMA 32 f32-FLOP/cycle/core",
    }
    for key in ("dram_gb_s", "peak_int8_gmacs", "peak_f32_gflops"):
        if not (isinstance(profile[key], (int, float)) and profile[key] > 0):
            raise SystemExit(f"ERROR: machine profile field {key} must be positive")
    return profile


class RooflineError(ValueError):
    """A floor cannot be computed honestly from the available inputs."""


# ── per-stage cost models (MACs + bytes; all terms are lower bounds) ─────────


def decode_per_token_cost(arch: str, precision: str, kv_len: int | None) -> dict:
    """Active weight traffic + MACs for ONE decode step (the G2 gating stage)."""
    if precision != "int8":
        raise RooflineError(f"decode floor modeled for int8 weights only, got {precision!r}")
    if arch == "unlimited-ocr":
        m = UNLIMITED
        h = m["hidden"]
        attn = m["layers"] * 4 * h * h
        dense0 = 3 * m["dense_inter"] * h
        moe = m["moe_layers"] * 3 * h * (m["experts_per_tok"] * m["moe_inter"] + m["shared_inter"])
        lm_head = m["vocab"] * h
        router_f32 = m["moe_layers"] * m["routed_experts"] * h  # f32 gate MACs
        int8_macs = attn + dense0 + moe + lm_head
        # int8 weight bytes = MAC count; per-out-channel f32 scales add 4B/row.
        scale_rows = (
            m["layers"] * 4 * h
            + (2 * m["dense_inter"] + h)
            + m["moe_layers"]
            * (
                m["experts_per_tok"] * (2 * m["moe_inter"] + h)
                + (2 * m["shared_inter"] + h)
            )
            + m["vocab"]
        )
        kv = m["kv_window"] if kv_len is None else min(kv_len, m["kv_window"])
        kv_bytes = m["layers"] * 2 * kv * h * 4  # f32 K+V ring reads
        kv_f32_macs = m["layers"] * 2 * kv * h
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * (router_f32 + kv_f32_macs),
            "bytes": int8_macs + scale_rows * 4 + router_f32 * 4 + kv_bytes,
            "assumptions": [
                "active set: attn qkvo + dense-L0 MLP + top-6 routed + fused shared expert + lm_head",
                "per-out-channel f32 scales counted; norms/embeds/activations excluded (lower bound)",
                f"R-SWA ring window {kv} K+V rows read per layer in f32",
            ],
        }
    if arch == "got-ocr2":
        m = GOT
        h = m["hidden"]
        int8_macs = m["layers"] * (4 * h * h + 3 * m["dense_inter"] * h) + m["vocab"] * h
        scale_rows = m["layers"] * (4 * h + 2 * m["dense_inter"] + h) + m["vocab"]
        if kv_len is None:
            raise RooflineError(
                "got-ocr2 decode attention is full-causal: kv_len must come from the "
                "measurement file (prefill + decoded tokens), not a guess"
            )
        kv_bytes = m["layers"] * 2 * kv_len * h * 4
        kv_f32_macs = m["layers"] * 2 * kv_len * h
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * kv_f32_macs,
            "bytes": int8_macs + scale_rows * 4 + kv_bytes,
            "assumptions": [
                "dense Qwen2: attn qkvo + SwiGLU MLP + lm_head, all int8",
                f"full-causal K+V read at mean context {kv_len} in f32",
                "norms/embeds/biases/activations excluded (lower bound)",
            ],
        }
    if arch in ("smolvlm2", "onechart"):
        m = SMOLVLM2 if arch == "smolvlm2" else ONECHART
        h, kvd = m["hidden"], m["kv_dim"]
        if arch == "smolvlm2":
            # GQA: q/o are h×h, k/v are h×kv_dim; SwiGLU = 3 GEMMs.
            gemm_per_layer = 2 * h * h + 2 * h * kvd + 3 * m["dense_inter"] * h
            scale_rows = m["layers"] * (2 * h + 2 * kvd + 2 * m["dense_inter"] + h)
            mlp_note = "SwiGLU gate/up/down"
        else:
            # OPT MHA qkvo + ReLU fc1/fc2 (2 GEMMs).
            gemm_per_layer = 4 * h * h + 2 * m["relu_inter"] * h
            scale_rows = m["layers"] * (4 * h + m["relu_inter"] + h)
            mlp_note = "ReLU fc1/fc2"
        int8_macs = m["layers"] * gemm_per_layer
        head_macs = m["vocab"] * h  # f32 head (untied-f32 / tied-embed)
        if kv_len is None:
            raise RooflineError(
                f"{arch} decode attention is full-causal: kv_len must come from the "
                "measurement file (prefill + decoded tokens), not a guess"
            )
        kv_bytes = m["layers"] * 2 * kv_len * kvd * 4
        kv_f32_macs = m["layers"] * 2 * kv_len * h  # scores + context over all q heads
        return {
            "int8_macs": int8_macs,
            "f32_flops": 2 * (head_macs + kv_f32_macs),
            "bytes": int8_macs + scale_rows * 4 + head_macs * 4 + kv_bytes,
            "assumptions": [
                f"dense {arch}: attn qkvo + {mlp_note} int8; the head is an f32 GEMM "
                "(model_arch.rs: lm_head_stored_int8 false / tied f32 embed)",
                f"full-causal K+V read at mean context {kv_len} in f32 (kv_dim {kvd})",
                "norms/embeds/biases/activations excluded (lower bound)",
            ],
        }
    raise RooflineError(f"unknown arch {arch!r}")


def prefill_cost(arch: str, precision: str, prefill_tokens: int) -> dict:
    """One prefill pass over `prefill_tokens` positions (lm_head excluded)."""
    if prefill_tokens <= 0:
        raise RooflineError("prefill floor needs the measured prefill token count")
    if precision != "int8":
        raise RooflineError(f"prefill floor modeled for int8 weights only, got {precision!r}")
    n = prefill_tokens
    if arch == "unlimited-ocr":
        m = UNLIMITED
        h = m["hidden"]
        per_tok = (
            m["layers"] * 4 * h * h
            + 3 * m["dense_inter"] * h
            + m["moe_layers"]
            * 3
            * h
            * (m["experts_per_tok"] * m["moe_inter"] + m["shared_inter"])
        )
        # Weight bytes at least once (identical-routing lower bound for MoE).
        weight_bytes = (
            m["layers"] * 4 * h * h
            + 3 * m["dense_inter"] * h
            + m["moe_layers"] * 3 * h * (m["experts_per_tok"] * m["moe_inter"] + m["shared_inter"])
        )
        attn_f32 = m["layers"] * 2 * n * n * h  # QK^T + AV, causal ~n²/2 rounded up
        return {
            "int8_macs": n * per_tok,
            "f32_flops": 2 * attn_f32,
            "bytes": weight_bytes,
            "assumptions": [
                f"{n} prefill tokens × active per-token GEMM MACs (lm_head excluded)",
                "memory floor = weights streamed once (identical-routing MoE lower bound)",
                "attention scores/context counted full-causal in f32",
            ],
        }
    if arch == "got-ocr2":
        m = GOT
        h = m["hidden"]
        per_tok = m["layers"] * (4 * h * h + 3 * m["dense_inter"] * h)
        attn_f32 = m["layers"] * 2 * n * n * h
        return {
            "int8_macs": n * per_tok,
            "f32_flops": 2 * attn_f32,
            "bytes": per_tok,
            "assumptions": [
                f"{n} prefill tokens × dense per-token GEMM MACs (lm_head excluded)",
                "memory floor = weights streamed once",
            ],
        }
    if arch in ("smolvlm2", "onechart"):
        m = SMOLVLM2 if arch == "smolvlm2" else ONECHART
        h, kvd = m["hidden"], m["kv_dim"]
        if arch == "smolvlm2":
            per_tok = m["layers"] * (2 * h * h + 2 * h * kvd + 3 * m["dense_inter"] * h)
        else:
            per_tok = m["layers"] * (4 * h * h + 2 * m["relu_inter"] * h)
        attn_f32 = m["layers"] * 2 * n * n * h
        return {
            "int8_macs": n * per_tok,
            "f32_flops": 2 * attn_f32,
            "bytes": per_tok,
            "assumptions": [
                f"{n} prefill tokens × dense per-token GEMM MACs (head excluded)",
                "memory floor = int8 weights streamed once",
            ],
        }
    raise RooflineError(f"unknown arch {arch!r}")


def vision_cost(arch: str, views: int) -> dict:
    """The f32 vision tower(s), per page = `views` × per-view GEMM terms."""
    if views <= 0:
        raise RooflineError("vision floor needs the measured view count (vision_sam occurrences)")
    if arch == "smolvlm2":
        # SigLIP-B/16 per 512² frame (views = frames), full attention, then the
        # pixel-shuffle connector GEMM (token_compress.rs).
        g = VISION_SIGLIP
        per_frame = (
            g["depth"] * g["tokens"] * (4 * g["dim"] * g["dim"] + 2 * g["dim"] * g["mlp"])
            + g["depth"] * 2 * g["tokens"] * g["tokens"] * g["dim"]
            + g["connector_macs"]
        )
        return {
            "int8_macs": 0,
            "f32_flops": views * 2 * per_frame,
            "bytes": g["params"] * 4,
            "assumptions": [
                f"SigLIP-B/16 blocks + full attention over 1024 tokens × {views} frames "
                "+ the 12288→960 modality_projection",
                "f32 weights streamed once; norms/softmax/pos-embeds excluded (lower bound)",
            ],
        }
    s = VISION_SAM
    sam_gemm = s["depth"] * s["tokens"] * (4 * s["dim"] * s["dim"] + 2 * s["dim"] * s["mlp"])
    sam_attn = s["tokens"] * s["dim"] * 2 * (
        (s["depth"] - s["global_blocks"]) * s["window"] + s["global_blocks"] * s["tokens"]
    )
    macs = sam_gemm + sam_attn + s["neck_convs_macs"]
    bytes_ = s["params"] * 4
    assumptions = ["SAM-ViT-B blocks + windowed/global attention + Vary neck/compressor convs"]
    if arch == "unlimited-ocr":
        c = VISION_CLIP
        macs += c["depth"] * c["tokens"] * (4 * c["dim"] * c["dim"] + 2 * c["dim"] * c["ffn"])
        macs += BRIDGE["tokens_per_view"] * BRIDGE["proj_in"] * BRIDGE["proj_out"]
        bytes_ += (
            c["depth"] * (4 * c["dim"] * c["dim"] + 2 * c["dim"] * c["ffn"])
            + BRIDGE["proj_in"] * BRIDGE["proj_out"]
        ) * 4
        assumptions.append("CLIP-L NoTP tower (24×1024/4096 over 257 tokens) + 2048→1280 bridge")
    elif arch == "got-ocr2":
        macs += BRIDGE["tokens_per_view"] * 1024 * 1024  # mm_projector_vary
        bytes_ += 1024 * 1024 * 4
        assumptions.append("GOT mm_projector_vary 1024→1024 over 256 tokens")
    elif arch == "onechart":
        macs += BRIDGE["tokens_per_view"] * 1024 * 768  # mm_projector 1024→768
        bytes_ += 1024 * 768 * 4
        assumptions.append("OneChart mm_projector 1024→768 over 256 tokens (same SAM tower)")
    else:
        raise RooflineError(f"unknown arch {arch!r}")
    assumptions.append("f32 weights streamed once; norms/softmax/pos-embeds excluded (lower bound)")
    return {
        "int8_macs": 0,
        "f32_flops": views * 2 * macs,
        "bytes": bytes_,  # weights dominate; activations excluded (lower bound)
        "assumptions": assumptions,
    }


def preprocess_cost(views: int) -> dict:
    """Producing the model input tensor(s) is the only safely boundable term."""
    if views <= 0:
        raise RooflineError("preprocess floor needs the measured view count")
    return {
        "int8_macs": 0,
        "f32_flops": 0,
        "bytes": views * 3 * 1024 * 1024 * 4,  # normalized f32 CHW tensor write per view
        "assumptions": [
            "memory floor only: the normalized 3×1024×1024 f32 input tensor must be written",
            "source decode/resize excluded (image dims not in the timing evidence)",
        ],
    }


def floor_from_cost(cost: dict, profile: dict) -> dict:
    compute_s = cost["int8_macs"] / (profile["peak_int8_gmacs"] * 1e9) + cost["f32_flops"] / (
        profile["peak_f32_gflops"] * 1e9
    )
    memory_s = cost["bytes"] / (profile["dram_gb_s"] * 1e9)
    kind = "compute" if compute_s >= memory_s else "memory"
    return {
        "floor_kind": kind,
        "floor_ms": round(max(compute_s, memory_s) * 1000.0, 6),
        "compute_floor_ms": round(compute_s * 1000.0, 6),
        "memory_floor_ms": round(memory_s * 1000.0, 6),
        "int8_macs": cost["int8_macs"],
        "f32_flops": cost["f32_flops"],
        "bytes": cost["bytes"],
        "assumptions": cost["assumptions"],
    }


# ── measurement-file plumbing ────────────────────────────────────────────────


def load_stages(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != "focr-gauntlet-stages/v1" or doc.get("source") != "focr":
        raise RooflineError(f"{path}: not a focr-gauntlet-stages/v1 focr measurement file")
    return doc


def stage_record(doc: dict, stage: str) -> dict | None:
    for record in doc.get("stages", []):
        if record.get("stage") == stage:
            return record
    return None


def measured_tokens(doc: dict, stage: str) -> int:
    record = stage_record(doc, stage)
    if record is None or not record.get("tokens"):
        raise RooflineError(f"measurement file has no token count for stage {stage!r}")
    return int(record["tokens"])


def measured_views(doc: dict) -> int:
    # SmolVLM2: the frame count rides the vision+splice record as `views`.
    record = stage_record(doc, "smolvlm2_vision_splice")
    if record is not None:
        views = record.get("views")
        if not views:
            raise RooflineError("smolvlm2_vision_splice record carries no frame count")
        return int(views)
    for name in ("vision_sam", "got_vision_splice", "onechart_vision_splice", "vision_encode"):
        record = stage_record(doc, name)
        if record is not None:
            return int(record.get("occurrences", 1))
    raise RooflineError("measurement file has no vision stage to derive the view count from")


def compute_stage_floor(stage: str, arch: str, precision: str, doc: dict, profile: dict) -> dict:
    if stage == "decode_per_token":
        kv_len = None
        if arch in ("got-ocr2", "smolvlm2", "onechart"):
            # Full-causal lanes: mean decode context = prefill + half the stream.
            prefill = measured_tokens(doc, "prefill") if _has_tokens(doc, "prefill") else 0
            decoded = measured_tokens(doc, "decode_total")
            kv_len = prefill + decoded // 2
            if prefill == 0:
                raise RooflineError(
                    f"{arch} decode floor needs the measured prefill token count"
                )
        cost = decode_per_token_cost(arch, precision, kv_len)
    elif stage == "prefill":
        cost = prefill_cost(arch, precision, measured_tokens(doc, "prefill"))
    elif stage == "vision_encode":
        cost = vision_cost(arch, measured_views(doc))
    elif stage == "preprocess":
        cost = preprocess_cost(measured_views(doc))
    elif stage == "end_to_end":
        parts = {}
        for sub in ("preprocess", "vision_encode", "prefill", "decode_per_token"):
            if (
                sub in ("preprocess",)
                and arch in ("got-ocr2", "smolvlm2", "onechart")
                and stage_record(doc, sub) is None
            ):
                continue  # zoo paths emit no separate preprocess line (vision+splice includes it)
            parts[sub] = compute_stage_floor(sub, arch, precision, doc, profile)
        decoded = measured_tokens(doc, "decode_total")
        floor_ms = sum(
            p["floor_ms"] * (decoded if name == "decode_per_token" else 1)
            for name, p in parts.items()
        )
        dominant = max(
            parts.items(),
            key=lambda kv: kv[1]["floor_ms"] * (decoded if kv[0] == "decode_per_token" else 1),
        )
        return {
            "floor_kind": dominant[1]["floor_kind"],
            "floor_ms": round(floor_ms, 6),
            "compute_floor_ms": None,
            "memory_floor_ms": None,
            "int8_macs": None,
            "f32_flops": None,
            "bytes": None,
            "assumptions": [
                f"sum of modeled stage floors with decode × {decoded} measured tokens",
                f"floor_kind = dominant component ({dominant[0]})",
            ],
            "components": parts,
        }
    else:
        raise RooflineError(f"no floor model for stage {stage!r}")
    return floor_from_cost(cost, profile)


def _has_tokens(doc: dict, stage: str) -> bool:
    record = stage_record(doc, stage)
    return record is not None and bool(record.get("tokens"))


def run(args: argparse.Namespace) -> int:
    profile = resolve_profile(args)
    doc = load_stages(args.stages_json)
    if bool(doc.get("synthetic")):
        # Floors over synthetic measurements are for plumbing tests only.
        print("WARNING: stages file is synthetic; output is stamped synthetic", file=sys.stderr)

    stages = args.stage or [
        s for s in STAGES if stage_record(doc, s) is not None or s == "decode_per_token"
    ]
    floors = []
    for stage in stages:
        try:
            floor = compute_stage_floor(stage, args.arch, args.precision, doc, profile)
        except RooflineError as err:
            print(f"ERROR: {stage}: {err}", file=sys.stderr)
            return 1
        record = stage_record(doc, stage)
        best_ms = record["best_ms"] if record else None
        floor["stage"] = stage
        floor["focr_best_ms"] = best_ms
        floor["dist_above_floor"] = (
            round(best_ms / floor["floor_ms"], 4) if best_ms and floor["floor_ms"] > 0 else None
        )
        floors.append(floor)

    out = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": args.arch,
        "precision": args.precision,
        "machine_profile": profile,
        "stages_json": args.stages_json,
        "floors": floors,
        "synthetic": bool(doc.get("synthetic")),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "roofline_written",
                "out": args.out,
                "floors": {f["stage"]: f["floor_ms"] for f in floors},
                "synthetic": out["synthetic"],
            }
        )
    )
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, **fields: object) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail", **fields}))
        if not ok:
            failures.append(name)

    # Independent arithmetic for the Unlimited decode active set (the numbers a
    # reviewer would compute by hand from the src constants).
    attn = 12 * 4 * 1280 * 1280  # 78_643_200
    dense0 = 3 * 1280 * 6848  # 26_296_320
    moe = 11 * (6 * 3 * 1280 * 896 + 3 * 1280 * 1792)  # 302_776_320
    lm_head = 129_280 * 1280  # 165_478_400
    expected_macs = attn + dense0 + moe + lm_head
    cost = decode_per_token_cost("unlimited-ocr", "int8", None)
    check("unlimited-decode-macs", cost["int8_macs"] == expected_macs, macs=expected_macs)
    check(
        "unlimited-decode-bytes-exceed-weights",
        cost["bytes"] > expected_macs,  # scales + router + KV ring on top of weights
    )

    profile = {"dram_gb_s": 120.0, "peak_int8_gmacs": 1126.4, "peak_f32_gflops": 563.2}
    floor = floor_from_cost(cost, profile)
    # ~573 MB active set at 120 GB/s ⇒ memory-bound, roughly 4.8 ms/token.
    check("unlimited-decode-memory-bound", floor["floor_kind"] == "memory")
    check(
        "unlimited-decode-floor-magnitude",
        4.0 < floor["floor_ms"] < 6.0,
        floor_ms=floor["floor_ms"],
    )
    # With an absurdly fast DRAM the same stage flips to compute-bound.
    flipped = floor_from_cost(cost, profile | {"dram_gb_s": 1e6})
    check("floor-kind-flips-with-profile", flipped["floor_kind"] == "compute")
    check(
        "compute-floor-consistent",
        math.isclose(flipped["compute_floor_ms"], floor["compute_floor_ms"]),
    )

    # GOT decode refuses without a measured context length (full causal).
    try:
        decode_per_token_cost("got-ocr2", "int8", None)
        check("got-refuses-unmeasured-kv", False)
    except RooflineError:
        check("got-refuses-unmeasured-kv", True)

    # Prefill refuses a fabricated token count; accepts a measured one.
    try:
        prefill_cost("unlimited-ocr", "int8", 0)
        check("prefill-refuses-no-tokens", False)
    except RooflineError:
        check("prefill-refuses-no-tokens", True)
    pf = prefill_cost("unlimited-ocr", "int8", 289)
    check("prefill-macs-scale-with-tokens", pf["int8_macs"] == 289 * (attn + dense0 + moe))

    # Vision floor: per-view scaling.
    v1 = vision_cost("unlimited-ocr", 1)
    v3 = vision_cost("unlimited-ocr", 3)
    check("vision-flops-scale-with-views", v3["f32_flops"] == 3 * v1["f32_flops"])
    check("vision-weights-once", v3["bytes"] == v1["bytes"])

    # int8 floors refuse a non-int8 precision rather than mislabeling the row.
    try:
        decode_per_token_cost("unlimited-ocr", "f32", None)
        check("refuses-f32-decode-model", False)
    except RooflineError:
        check("refuses-f32-decode-model", True)

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-roofline-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--arch",
        choices=("unlimited-ocr", "got-ocr2", "smolvlm2", "onechart"),
        default="unlimited-ocr",
    )
    parser.add_argument("--precision", choices=("int8",), default="int8")
    parser.add_argument("--stages-json", default=None, help="focr_stages.json (required)")
    parser.add_argument("--stage", action="append", choices=STAGES, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="m4")
    parser.add_argument("--dram-gb-s", type=float, default=None)
    parser.add_argument("--peak-int8-gmacs", type=float, default=None)
    parser.add_argument("--peak-f32-gflops", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not args.stages_json or not args.out:
        parser.error("--stages-json and --out are required (floors bind to real measurements)")
    try:
        return run(args)
    except (RooflineError, OSError, json.JSONDecodeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
