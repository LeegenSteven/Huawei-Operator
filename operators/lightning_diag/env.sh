#!/usr/bin/env bash
# Source on the Ascend server; settings affect the current shell only.
export LIGHTNING_PYTHON_ROOT="${LIGHTNING_PYTHON_ROOT:-/usr/local/python3.11}"
export LIGHTNING_GCC_ROOT="${LIGHTNING_GCC_ROOT:-/opt/kunpeng-gcc}"
export LIGHTNING_CANN_ROOT="${LIGHTNING_CANN_ROOT:-/usr/local/Ascend/cann}"
export PATH="$LIGHTNING_PYTHON_ROOT/bin:$LIGHTNING_GCC_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$LIGHTNING_GCC_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The vendor loader reads optional, unset variables; preserve the caller's -u.
_LIGHTNING_RESTORE_NOUNSET=0
case "$-" in *u*) _LIGHTNING_RESTORE_NOUNSET=1; set +u ;; esac
source "$LIGHTNING_CANN_ROOT/set_env.sh"
if [ "$_LIGHTNING_RESTORE_NOUNSET" = 1 ]; then set -u; fi
unset _LIGHTNING_RESTORE_NOUNSET
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
