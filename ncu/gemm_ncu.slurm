#!/bin/bash
#SBATCH --job-name=ncu_gemm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=results/ncu_%j.out
#SBATCH --error=results/ncu_%j.err
# Launch with: submit_gemm_ncu.sh [index]

set -e

# Use submit directory so config and outputs are in the same place as the script
# (SLURM runs the job from the spool dir; BASH_SOURCE would point there)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

# submit_gemm_ncu.sh passes M/N/K directly.
if [[ -z "${M:-}" || -z "${N:-}" || -z "${K:-}" ]]; then
  echo "Missing M/N/K env vars. Use submit_gemm_ncu.sh to submit jobs." >&2
  exit 1
fi
if [[ ! "$M" =~ ^[0-9]+$ || ! "$N" =~ ^[0-9]+$ || ! "$K" =~ ^[0-9]+$ ]]; then
  echo "Invalid M/N/K env vars: M=$M N=$N K=$K" >&2
  exit 1
fi
TASK_INDEX="${CONFIG_INDEX:-NA}"
m="$M"; n="$N"; k="$K"
echo "Task ${TASK_INDEX}: m=$m n=$n k=$k"

module purge
module load cudatoolkit/12.8
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_INSTALL_PATH:$PATH

# ncu -o gemm_test_${SLURM_JOBID} --set=regression --target-processes=all \
#     /scratch/gpfs/WENTZLAF/hm2595/differential_arch/simulators/AccelSim/gpu-app-collection/bin/12.8/release/basic_gemm \
#     --seed=2020 --dist=0 --operation=gemm --m=2048 --n=2048 --k=2048 --op_class=tensorop --iterations=10 --provider=cutlass > gemm_test_${SLURM_JOBID}.log

# ncu -o gemm_test_${SLURM_JOBID} \
#     --target-processes=application-only \
#     --launch-skip 10 \
#     --launch-count 1 \
#     --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__cycles_elapsed.avg \
#     /scratch/gpfs/WENTZLAF/hm2595/differential_arch/simulators/AccelSim/gpu-app-collection/bin/12.8/release/basic_gemm \
#     --seed=2020 --dist=0 --operation=gemm \
#     --m=2048 --n=2048 --k=2048 \
#     --op_class=tensorop --iterations=10 --provider=cutlass

METRICS="sm__pipe_tensor_active.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_pipe_tensor,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_cycles_active,smsp__cycles_elapsed"
# METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__cycles_elapsed.avg,sm__pipe_tensor_active.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_pipe_tensor,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.sum,smsp__cycles_elapsed,smsp__inst_executed_pipe_tensor_op_hmma.sum,smsp__inst_executed_pipe_tensor.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__pipe_tensor_cycles_active,smsp__pipe_tensor_op_hmma_cycles_active,smsp__issue_active,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct"

BASENAME="gemm_bottleneck_${m}_${n}_${k}_${SLURM_JOB_ID}"

ncu -o "results/${BASENAME}" \
  --target-processes=application-only \
  --metrics "$METRICS" \
  /scratch/gpfs/WENTZLAF/hm2595/differential_arch/simulators/AccelSim/gpu-app-collection/bin/12.8/release/gemm_streamk \
  --seed=2020 --dist=0 --operation=gemm --m=$m --n=$n --k=$k \
  --op_class=tensorop --iterations=10 --provider=cutlass

cd results
ncu --import "${BASENAME}.ncu-rep" --csv --page raw > "${BASENAME}.csv"

python3 "${SCRIPT_DIR}/extract_ncu_metrics.py" "${BASENAME}.csv" --summary

echo "Done: m=$m n=$n k=$k -> ${BASENAME}.csv"
