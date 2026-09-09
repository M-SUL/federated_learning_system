#!/usr/bin/env bash
# Prepare the dataset and Flower config, then serve the console.
#
# Everything here is idempotent: on a restart with a populated volume it
# re-verifies in seconds rather than re-downloading ~200 MB.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
DATASET_DIR="${DATA_ROOT}/TCGA-PANCAN-HiSeq-801x20531"
RESULTS_DIR="${RESULTS_DIR:-/app/fl_rnaseq/results}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "== Federation console =="

# --- 1. Dataset ------------------------------------------------------------
# Downloaded on first boot rather than baked into the image. The pruning step
# (dropping 267 zero-variance genes) lived only in the EDA notebook, so
# prepare_data.py reproduces it headlessly and verifies the 801x20264 shape --
# a stale or partial file would otherwise surface much later as a confusing
# matrix-shape error inside a training round.
echo "-- dataset"
cd /app/fl_rnaseq
python -m scripts.prepare_data --data-dir "${DATA_ROOT}" 2>&1 | sed 's/^/   /' \
  || { echo "   dataset preparation FAILED"; exit 1; }

# pyproject.toml points at "../gene+expression+cancer+rna+seq/..." relative to the
# app directory, which is how it resolves on a developer's checkout. Rather than
# rewrite that config for the container (and have the two drift), make the same
# relative path land on the mounted volume.
LINK="/app/gene+expression+cancer+rna+seq"
if [ ! -e "${LINK}" ]; then
  ln -s "${DATA_ROOT}" "${LINK}"
  echo "   linked ${LINK} -> ${DATA_ROOT}"
fi
[ -f "${DATASET_DIR}/data_pruned.csv" ] || { echo "   missing ${DATASET_DIR}/data_pruned.csv"; exit 1; }

# --- 2. Flower config ------------------------------------------------------
# The SuperLink reads num-supernodes from here at startup, and the launcher
# rewrites this one line when the site count changes. Flower would create a
# default on first run, but the launcher needs the line to exist to edit it.
FLWR_CONFIG="${HOME:-/root}/.flwr/config.toml"
if [ ! -f "${FLWR_CONFIG}" ]; then
  echo "-- writing ${FLWR_CONFIG}"
  mkdir -p "$(dirname "${FLWR_CONFIG}")"
  cat > "${FLWR_CONFIG}" <<'TOML'
[superlink]
default = "local-simulation"

[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 5
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.0
TOML
else
  echo "-- flower config present: ${FLWR_CONFIG}"
fi

mkdir -p "${RESULTS_DIR}/live"

# --- 3. Launch policy ------------------------------------------------------
# serve.py refuses to start when launching is enabled on a public address with
# no password. Surface that here rather than letting it look like a crash loop.
LAUNCH_ARGS=()
if [ "${ALLOW_LAUNCH:-true}" = "true" ]; then
  LAUNCH_ARGS+=(--allow-remote-launch)
  if [ -z "${DASHBOARD_PASSWORD:-}" ]; then
    echo
    echo "   ERROR: ALLOW_LAUNCH=true but DASHBOARD_PASSWORD is unset."
    echo "   A public instance that can start training runs must require a password."
    echo "   Set DASHBOARD_PASSWORD, or set ALLOW_LAUNCH=false for read-only."
    echo
    exit 2
  fi
else
  LAUNCH_ARGS+=(--no-launch)
fi

echo "-- serving on ${HOST}:${PORT}"
exec python /app/webapp/serve.py \
  --host "${HOST}" --port "${PORT}" \
  --app-dir /app/fl_rnaseq \
  --results-dir "${RESULTS_DIR}" \
  "${LAUNCH_ARGS[@]}"
