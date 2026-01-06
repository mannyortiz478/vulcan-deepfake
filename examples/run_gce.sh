#!/usr/bin/env bash
# Example helper script to run training on a GCP VM with GPU.
# Usage: ./examples/run_gce.sh /path/to/asv --config configs/training/whisper_specrnet.yaml

set -euo pipefail

ASV_PATH=${1:-/path/to/asv}
CONFIG=${2:-configs/training/whisper_specrnet.yaml}
DEVICE=${3:-cuda}
BATCH=${4:-8}
EPOCHS=${5:-10}
ACCUM=${6:-1}
AMP_FLAG=${7:---amp}
TQDM_FLAG=${8:-}
LOGFILE=${9:-train_run.log}

# Activate virtualenv if exists
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

echo "Starting training. Logs: ${LOGFILE}"
nohup python train_and_test.py --asv_path "${ASV_PATH}" --config "${CONFIG}" --device "${DEVICE}" --batch_size ${BATCH} --epochs ${EPOCHS} --accum-steps ${ACCUM} ${AMP_FLAG} ${TQDM_FLAG} > ${LOGFILE} 2>&1 &

echo "Launched training in background; use 'tail -f ${LOGFILE}' to follow logs"