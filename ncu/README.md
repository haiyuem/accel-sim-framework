# NCU GEMM profiling

Standalone workflow that profiles cutlass-streamk GEMMs on real hardware with Nsight Compute, then clusters the per-kernel results by prefill/decode and by bottleneck. **Unrelated to the AccelSim simulator pipeline** — it only reuses the `gemm_streamk` binary built in `gpu-app-collection/`.

Output is the paper plot `prefill_decode_gemm_clusters.pdf` and the companion CSV `prefill_decode_component_summary.csv`.

## Pipeline

```bash
# 1. Regenerate spec CSVs from the model dims hard-coded in generate_gemm_configs.py
python3 generate_gemm_configs.py
#   writes gemm_specs_full.csv   (one row per model × phase × component)
#          gemm_specs_unique.csv (deduped on (M,N,K), with merged label)

# 2. Submit ncu profiling jobs on Della (needs a GPU node)
./submit_gemm_ncu.sh              # submit all rows in gemm_specs_unique.csv
./submit_gemm_ncu.sh 12           # submit just index 12
./submit_gemm_ncu.sh 5-8 --qos=debug

# 3. After jobs finish, run the notebook
#    prefill_decode_gemm_clusters.ipynb
#    → prefill_decode_gemm_clusters.pdf
#    → prefill_decode_component_summary.csv
```

`submit_gemm_ncu.sh` passes `M`, `N`, `K`, and `CONFIG_INDEX` as env vars to `gemm_ncu.slurm`. The slurm script runs `ncu -o results/gemm_bottleneck_<M>_<N>_<K>_<jobid>` with the metric set defined inline, converts the `.ncu-rep` to a CSV with `ncu --import … --page raw`, and calls `extract_ncu_metrics.py --summary` for a quick dram/lts/tensor-util printout.

## File roles

| File | Role |
| --- | --- |
| `generate_gemm_configs.py` | Source of truth for model dimensions. Writes `gemm_specs_{full,unique}.csv`. |
| `submit_gemm_ncu.sh` | Submits one sbatch per `gemm_specs_unique.csv` row. |
| `gemm_ncu.slurm` | Per-job runner: `ncu → .ncu-rep → .csv → summary`. Writes into `results/`. |
| `extract_ncu_metrics.py` | Parses NCU per-kernel CSVs into (dram%, lts%, tensor_util). Called by slurm and importable. |
| `prefill_decode_gemm_clusters.ipynb` | Walks `results/ncu_*.out`, joins against `gemm_specs_full.csv`, emits PDF + summary CSV. |
| `ln.sh` | One-shot: symlinks `gemm_streamk` under 24 per-(model, phase, component) names expected by the AccelSim sweep side (`apps/define-all-apps.yml`). Run once per fresh `gpu-app-collection` checkout. |

## Conventions

- Every artifact in `results/` is named `gemm_bottleneck_<M>_<N>_<K>_<slurm_jobid>.{csv,ncu-rep}`; slurm logs are `ncu_<jobid>_<array_idx>.{out,err}`. The notebook's regex relies on this format — don't rename.
- The notebook keys on `gemm_specs_full.csv` — MNK shapes present in `results/` but absent from the spec are silently ignored.
- Metric set is edited inline in `gemm_ncu.slurm` (`METRICS=…`). If you add a metric, rerun all shapes you care about — NCU can't backfill.
