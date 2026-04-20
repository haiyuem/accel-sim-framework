#!/bin/bash
# Submit NCU jobs in gemm_specs_unique.csv sequentially (one sbatch per index).
# Usage:
#   ./submit_gemm_ncu.sh [index|start-end] [sbatch options...]
# Examples:
#   ./submit_gemm_ncu.sh                 # submit all
#   ./submit_gemm_ncu.sh 25-31           # submit range 25..31
#   ./submit_gemm_ncu.sh 12 --qos=debug  # submit only index 12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGFILE="${SCRIPT_DIR}/gemm_specs_unique.csv"
RESULTS_DIR="${SCRIPT_DIR}/results"
if [[ ! -f "$CONFIGFILE" ]]; then
  echo "Missing $CONFIGFILE" >&2
  exit 1
fi
mkdir -p "$RESULTS_DIR"

# Count valid CSV rows with numeric M,N,K in the first three columns.
N=$(awk -F',' '
NF >= 3 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { c++ }
END { print c + 0 }
' "$CONFIGFILE")
N=$((N - 1))   # 0-based last index
if [[ $N -lt 0 ]]; then
  echo "No configs in $CONFIGFILE" >&2
  exit 1
fi

START_INDEX=0
END_INDEX="$N"
if [[ $# -gt 0 ]]; then
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    START_INDEX="$1"
    END_INDEX="$1"
    shift
  elif [[ "$1" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    START_INDEX="${BASH_REMATCH[1]}"
    END_INDEX="${BASH_REMATCH[2]}"
    shift
  fi
fi

if [[ "$START_INDEX" -gt "$END_INDEX" ]]; then
  echo "Invalid range: ${START_INDEX}-${END_INDEX}" >&2
  exit 1
fi
if [[ "$END_INDEX" -gt "$N" ]]; then
  echo "Range end ${END_INDEX} out of range, max index is ${N}" >&2
  exit 1
fi

TOTAL=$((END_INDEX - START_INDEX + 1))
echo "Submitting ${TOTAL} jobs (index ${START_INDEX}..${END_INDEX}, max=${N})"

for INDEX in $(seq "$START_INDEX" "$END_INDEX"); do
  # Convert 0-based index to 1-based line number within valid MNK rows.
  ROW_NUM=$((INDEX + 1))
  MNK_LINE=$(awk -F',' '
NF >= 3 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { print $1 " " $2 " " $3 }
' "$CONFIGFILE" | sed -n "${ROW_NUM}p")
  read -r m n k <<< "$MNK_LINE"
  if [[ -z "${m:-}" || -z "${n:-}" || -z "${k:-}" ]]; then
    echo "Failed to resolve M/N/K for index ${INDEX} from $CONFIGFILE" >&2
    exit 1
  fi

  echo "Submitting config index ${INDEX}/${N}: m=${m} n=${n} k=${k}"
  if ! sbatch --export=ALL,CONFIG_INDEX="${INDEX}",M="${m}",N="${n}",K="${k}" "$@" "${SCRIPT_DIR}/gemm_ncu.slurm"; then
    echo "Submission failed at index ${INDEX}; stopping." >&2
    exit 1
  fi
done
