#!/usr/bin/env python3
"""Score franken_ocr output against the baidu reference (the parity verdict).

Compares two directories of per-page markdown (`page_XXXX.md`):
  --ref  : baidu/Unlimited-OCR reference  (run_baidu_reference.py output)
  --hyp  : franken_ocr native output      (`focr ocr page.png`)

Reports per-page and aggregate Character Error Rate (CER, Levenshtein/len(ref))
and exact-match, with both raw and whitespace-normalized variants. CER is the
standard OCR quality metric; for a faithful port we expect aggregate CER well
under a small threshold (and ideally exact text on clean pages).

Usage: compare_ocr.py --ref DIR --hyp DIR [--normalize] [--json OUT.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path


def levenshtein(a: str, b: str) -> int:
    """Edit distance with a two-row DP (O(len(a)*len(b)) time, O(len(b)) space)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def norm_ws(s: str) -> str:
    """Collapse runs of whitespace and strip — tolerates layout-only differences."""
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ref_dir, hyp_dir = Path(args.ref), Path(args.hyp)
    ref_pages = sorted(ref_dir.glob("page_*.md"))
    if not ref_pages:
        print(f"no reference pages in {ref_dir}", file=sys.stderr)
        sys.exit(2)

    rows = []
    tot_raw_ed = tot_raw_len = 0
    tot_norm_ed = tot_norm_len = 0
    n_exact = n_exact_norm = n_have_hyp = 0
    for rp in ref_pages:
        hp = hyp_dir / rp.name
        ref = rp.read_text()
        if not hp.exists():
            rows.append({"page": rp.name, "status": "MISSING_HYP",
                         "ref_chars": len(ref)})
            tot_raw_len += len(ref)
            tot_norm_len += len(norm_ws(ref))
            continue
        n_have_hyp += 1
        hyp = hp.read_text()
        raw_ed = levenshtein(ref, hyp)
        nref, nhyp = norm_ws(ref), norm_ws(hyp)
        norm_ed = levenshtein(nref, nhyp)
        tot_raw_ed += raw_ed
        tot_raw_len += len(ref)
        tot_norm_ed += norm_ed
        tot_norm_len += len(nref)
        exact = ref == hyp
        exact_norm = nref == nhyp
        n_exact += exact
        n_exact_norm += exact_norm
        rows.append({
            "page": rp.name,
            "status": "OK",
            "ref_chars": len(ref),
            "hyp_chars": len(hyp),
            "cer_raw": round(raw_ed / max(1, len(ref)), 5),
            "cer_norm": round(norm_ed / max(1, len(nref)), 5),
            "exact": exact,
            "exact_norm": exact_norm,
        })

    agg = {
        "pages_total": len(ref_pages),
        "pages_with_hyp": n_have_hyp,
        "exact_raw": n_exact,
        "exact_norm": n_exact_norm,
        "cer_raw": round(tot_raw_ed / max(1, tot_raw_len), 5),
        "cer_norm": round(tot_norm_ed / max(1, tot_norm_len), 5),
    }

    print(f"{'page':16} {'status':12} {'ref':>6} {'hyp':>6} {'CER_raw':>8} {'CER_norm':>8} {'exact':>6}")
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['page']:16} {r['status']:12} {r['ref_chars']:>6}")
            continue
        print(f"{r['page']:16} {r['status']:12} {r['ref_chars']:>6} {r['hyp_chars']:>6} "
              f"{r['cer_raw']:>8.4f} {r['cer_norm']:>8.4f} {str(r['exact']):>6}")
    print("-" * 72)
    print(f"AGGREGATE: pages={agg['pages_total']} with_hyp={agg['pages_with_hyp']} "
          f"exact={agg['exact_raw']} exact_norm={agg['exact_norm']} "
          f"CER_raw={agg['cer_raw']:.4f} CER_norm={agg['cer_norm']:.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps({"aggregate": agg, "pages": rows}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
