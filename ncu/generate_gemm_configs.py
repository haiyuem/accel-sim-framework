#!/usr/bin/env python3
"""Generate GEMM configs and metadata for NCU runs.

This script mirrors the specs used in prefill_decode_gemm_clusters.ipynb and writes:
  1) gemm_specs_full.csv    (one row per model/phase/component entry)
  2) gemm_specs_unique.csv  (unique MNK with merged labels)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Per-model dimensions (edit if needed)
DEEPSEEK_D = 7168
DEEPSEEK_DFF = 18432

LLAMA_D = 8192
LLAMA_DFF = 28672

# Mixtral-8x7B (MoE) dense GEMM dimensions
MIXTRAL_D = 4096
MIXTRAL_DFF = 14336

# Batch/token settings used in this workflow
# Decode: query is current-step tokens, KV is cache length.
# Prefill: query and KV lengths are both prompt/context length.
M_QUERY_DECODE = 128
M_QUERY_DECODE_MIXTRAL = 128
M_KV_DECODE = 4096
M_QUERY_PREFILL = 4096
M_KV_PREFILL = 4096


def build_specs() -> pd.DataFrame:
    specs: list[dict] = []

    def add_model_specs(
        model_name: str, d: int, dff: int, m_query_decode: int = M_QUERY_DECODE
    ) -> None:
        qkv_n = 3 * d
        ffn_up_n = 2 * dff  # SwiGLU combined up/gate width

        # decode
        specs.extend(
            [
                {
                    "model": model_name,
                    "phase": "decode",
                    "component": "qkv",
                    "M": m_query_decode,
                    "N": qkv_n,
                    "K": d,
                },
                {
                    "model": model_name,
                    "phase": "decode",
                    "component": "atten_out",
                    "M": m_query_decode,
                    "N": d,
                    "K": d,
                },
                {
                    "model": model_name,
                    "phase": "decode",
                    "component": "ffn_up",
                    "M": m_query_decode,
                    "N": ffn_up_n,
                    "K": d,
                },
                {
                    "model": model_name,
                    "phase": "decode",
                    "component": "ffn_down",
                    "M": m_query_decode,
                    "N": d,
                    "K": dff,
                },
            ]
        )

        # prefill
        specs.extend(
            [
                {
                    "model": model_name,
                    "phase": "prefill",
                    "component": "qkv",
                    "M": M_QUERY_PREFILL,
                    "N": qkv_n,
                    "K": d,
                },
                {
                    "model": model_name,
                    "phase": "prefill",
                    "component": "atten_out",
                    "M": M_QUERY_PREFILL,
                    "N": d,
                    "K": d,
                },
                {
                    "model": model_name,
                    "phase": "prefill",
                    "component": "ffn_up",
                    "M": M_QUERY_PREFILL,
                    "N": ffn_up_n,
                    "K": d,
                },
                {
                    "model": model_name,
                    "phase": "prefill",
                    "component": "ffn_down",
                    "M": M_QUERY_PREFILL,
                    "N": d,
                    "K": dff,
                },
            ]
        )

    add_model_specs("DeepSeek-V3", DEEPSEEK_D, DEEPSEEK_DFF)
    add_model_specs("Llama-3-70B", LLAMA_D, LLAMA_DFF)
    add_model_specs(
        "Mixtral-8x7B", MIXTRAL_D, MIXTRAL_DFF, m_query_decode=M_QUERY_DECODE_MIXTRAL
    )

    return pd.DataFrame(specs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate GEMM spec CSVs from target model specs")
    ap.add_argument("--out-dir", default=".", help="Output directory (default: current dir)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_df = build_specs()
    full_csv = out_dir / "gemm_specs_full.csv"
    unique_csv = out_dir / "gemm_specs_unique.csv"
    full = spec_df.copy()
    full["label"] = full["model"] + "/" + full["phase"] + "/" + full["component"]
    grouped = (
        full.groupby(["M", "N", "K"], as_index=False)
        .agg(label=("label", lambda x: "; ".join(sorted(set(x)))))
        .sort_values(["M", "K", "N"])
    )

    spec_df = spec_df.sort_values(["model", "phase", "component"]).reset_index(drop=True)
    spec_df.to_csv(full_csv, index=False)

    grouped.to_csv(unique_csv, index=False)

    print(f"wrote: {full_csv}")
    print(f"wrote: {unique_csv}")
    print(f"full specs rows: {len(spec_df)}")
    print(f"unique MNK rows: {len(grouped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
