#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_DIR="${1:-E:/ClinicalICG/WhiteLight2ICGVideo/code/4-WhiteLight2ICGVideo-ConvGRU/experiments/261011-ablation_261009_no_time}"
OUT_DIR="${2:-./outputs/best_rgb2vessel_mask}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
if [[ ! -f "${EXPERIMENT_DIR}/checkpoints/best.pt" ]]; then
  echo "Missing checkpoint: ${EXPERIMENT_DIR}/checkpoints/best.pt" >&2
  exit 1
fi
if [[ -d "${OUT_DIR}" ]] && [[ -n "$(ls -A "${OUT_DIR}")" ]]; then
  echo "Output directory is not empty. Supply a new second argument: ${OUT_DIR}" >&2
  exit 1
fi
EXPERIMENTS_ROOT="$(dirname "${EXPERIMENT_DIR}")"
RUN_NAME="$(basename "${EXPERIMENT_DIR}")"
"${PYTHON_BIN}" infer_patch.py --experiments_root "${EXPERIMENTS_ROOT}" \
  --experiment_name "${RUN_NAME}" --checkpoint_type best --splits test \
  --input_vessel_mask_source rgb2vessel_mask --save_panels --save_targets --out_dir "${OUT_DIR}"
"${PYTHON_BIN}" notebook/summarize_experiment_metrics_by_split.py \
  --experiment "${RUN_NAME}" --experiments-root "${EXPERIMENTS_ROOT}" \
  --mode "$(basename "${OUT_DIR}")" --inference-dir "$(dirname "${OUT_DIR}")" \
  --supplement-only --summary-splits test --out-dir "${OUT_DIR}/metrics_by_split"
