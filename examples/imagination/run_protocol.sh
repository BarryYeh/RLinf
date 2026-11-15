#!/bin/bash
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One run of the imagination SFT protocol (README "Protocol"), resumable.
#
#   real     the base policy on the imagination states, K reruns
#   anchors  with ANCHOR_METHOD set, start points shortly before each real failure
#   imagine  N closed-loop rollouts per state in the world model, PARTS in parallel
#   judge    Robometer success calls on the imagined rollouts
#   export   per task, the top KEEP judged successes per state, cut at success
#   sft      demos only, and demos + imagined at 50/50 (equal weight per task)
#   eval     base, demos, imagined on the test states, 2 reruns
#   monitor  open-loop replay of the validation run, judged on both sides
#   report   report.md / report.json
#
# Usage: run_protocol.sh <run_dir>. Settings are the environment variables below;
# the first run records them in <run_dir>/protocol.json, and a later run with
# other settings refuses to start. A finished stage leaves <run_dir>/<stage>.done.
set -uo pipefail

RUN=$(realpath -m "${1:?usage: run_protocol.sh <run_dir>}")
REPO=$(realpath "$(dirname "$0")/../..")
DS=$HOME/datastorage

BASE_CKPT=${BASE_CKPT:-$DS/runs/rlinf/20261102-multitask_dit_sweep/ep10/libero_spatial_sft_multitask_dit_ep10/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt}
MODEL_CFG=${MODEL_CFG:-$REPO/examples/embodiment/config/model/multitask_dit.yaml}
WM_CKPT=${WM_CKPT:-$DS/models/RLinf-OpenSora-LIBERO-Spatial}
STATES=${STATES:-$DS/datasets/libero_spatial_states}
DEMOS=${DEMOS:-$DS/datasets/libero}
DEMOS_PER_TASK=${DEMOS_PER_TASK:-10}
VAL_REAL=${VAL_REAL:-$DS/runs/imagination/real_val_dit_ep10_30k_k4}
K=${K:-4}
N=${N:-8}
KEEP=${KEEP:-4}
ANCHOR_STEP=${ANCHOR_STEP:-0}
ANCHOR_METHOD=${ANCHOR_METHOD:-}  # failure-anchor score (select_anchors.py --method), empty = ANCHOR_STEP
ANCHOR_ARGS=${ANCHOR_ARGS:-}      # more select_anchors.py options, e.g. "--ace-samples 32"
SEED=${SEED:-0}
PARTS=${PARTS:-4}
TRIALS=${TRIALS:-}            # imagination trials per task, empty = all
TEST_TRIALS=${TEST_TRIALS:-}  # test trials per task, empty = all 50
TEST_RERUNS=${TEST_RERUNS:-2}
SFT_STEPS=${SFT_STEPS:-10000}
MONITOR=${MONITOR:-1}
BASE_EVAL=${BASE_EVAL:-}      # a finished test eval of BASE_CKPT to reuse, empty = run it
MAIN_VENV=${MAIN_VENV:-$REPO/.venv}
OPENSORA_VENV=${OPENSORA_VENV:-$DS/venvs/opensora}
ROBOMETER_VENV=${ROBOMETER_VENV:-$DS/venvs/robometer}
export HF_HOME=${HF_HOME:-$DS/cache/huggingface}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$REPO
CACHE_DIR=${CACHE_DIR:-$DS/cache/multitask_dit}
SETTINGS=(BASE_CKPT MODEL_CFG WM_CKPT STATES DEMOS DEMOS_PER_TASK VAL_REAL K N KEEP
  ANCHOR_STEP SEED PARTS TRIALS TEST_TRIALS TEST_RERUNS SFT_STEPS MONITOR BASE_EVAL)
# Recorded only when set, so run dirs from before the setting still resume.
[ -z "$ANCHOR_METHOD" ] || SETTINGS+=(ANCHOR_METHOD ANCHOR_ARGS)

IMAGINATION=$REPO/examples/imagination
PY=$MAIN_VENV/bin/python
mkdir -p "$RUN"
cd "$REPO"
log() { echo "$(date '+%F %T') $*" | tee -a "$RUN/pipeline.log"; }
fail() { log "FAILED: $*"; exit 1; }

# Records the settings on the first run and refuses a mismatch later.
settings=$(for name in "${SETTINGS[@]}"; do printf '%s\t%s\n' "$name" "${!name}"; done)
SETTINGS_TSV="$settings" GIT_HEAD=$(git rev-parse HEAD) GIT_DIFF=$(git diff --stat HEAD | tail -1) \
  $PY - "$RUN/protocol.json" <<'EOF' || exit 1
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
settings = dict(line.split("\t", 1) for line in os.environ["SETTINGS_TSV"].splitlines())
if path.exists():
    recorded = json.loads(path.read_text())["settings"]
    if recorded != settings:
        diff = {k: (recorded.get(k), v) for k, v in settings.items() if recorded.get(k) != v}
        sys.exit(f"Expected the settings recorded in {path}, found changes {diff}; use a new run dir.")
else:
    git = {"head": os.environ["GIT_HEAD"], "diff_stat": os.environ["GIT_DIFF"]}
    path.write_text(json.dumps({"settings": settings, "git": git}, indent=2))
EOF

# Runs "$@" as stage $1 unless it is done.
stage() {
  local name=$1
  shift
  if [ -f "$RUN/$name.done" ]; then
    log "$name: done before, skipping"
    return 0
  fi
  log "$name: start"
  "$@" || fail "$name"
  touch "$RUN/$name.done"
  log "$name: done"
}

judge() {
  "$ROBOMETER_VENV/bin/python" "$IMAGINATION/robometer_judge.py" --rollouts "$1" > "$1.judge.log" 2>&1
}

collect() {  # collect <ckpt> <out> <collect_rollouts args...>
  local ckpt=$1 out=$2
  shift 2
  $PY "$IMAGINATION/collect_rollouts.py" --ckpt "$ckpt" --model-cfg "$MODEL_CFG" \
    --out "$out" --ignore-terminations --seed "$SEED" "$@" > "$out.log" 2>&1
}

real() {
  collect "$BASE_CKPT" "$RUN/real" --init-states-dir "$STATES/imagination" \
    --reruns "$K" ${TRIALS:+--trials "$TRIALS"}
}

select_anchors() {
  local extra
  read -ra extra <<< "$ANCHOR_ARGS"
  $PY "$IMAGINATION/select_anchors.py" --method "$ANCHOR_METHOD" --real "$RUN/real" \
    --ckpt "$BASE_CKPT" --model-cfg "$MODEL_CFG" --out "$RUN/anchors.json" \
    --rnd-demos "$DEMOS" --rnd-val-run "$VAL_REAL" --rnd-cache-dir "$CACHE_DIR" \
    --seed "$SEED" "${extra[@]}" > "$RUN/anchors.log" 2>&1
}

imagine() {
  local trials p pids=() start=(--anchor-step "$ANCHOR_STEP")
  [ -n "$ANCHOR_METHOD" ] && start=(--anchors "$RUN/anchors.json")
  # The real run's trial ids, split into PARTS contiguous ranges of near-equal size.
  read -ra trials <<< "$($PY -c "
import sys; sys.path.insert(0, '$IMAGINATION')
from common import load_records
ids = sorted({r['trial_id'] for r in load_records('$RUN/real')})
parts = min($PARTS, len(ids))
size, extra = divmod(len(ids), parts)
start = 0
for p in range(parts):
    end = start + size + (p < extra)
    print(','.join(map(str, ids[start:end])), end=' ')
    start = end
")"
  mkdir -p "$RUN/imagined"
  for p in "${!trials[@]}"; do
    "$OPENSORA_VENV/bin/python" "$IMAGINATION/imagine_rollouts.py" --mode policy \
      --real "$RUN/real" --wm-ckpt "$WM_CKPT" --ckpt "$BASE_CKPT" --model-cfg "$MODEL_CFG" \
      --out "$RUN/imagined/part_$p" --trials "${trials[$p]}" --reruns "$N" \
      "${start[@]}" --num-envs 10 --seed $((SEED + p)) --no-fused-kernels \
      > "$RUN/imagined/part_$p.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || return 1; done
  mkdir -p "$RUN/imagined/all/episodes"
  ln -sf "$RUN"/imagined/part_*/episodes/*.npz "$RUN/imagined/all/episodes/"
}

export_kept() {
  local t pids=()
  rm -rf "$RUN/lerobot_imagined"
  for t in $(seq 0 9); do
    $PY "$IMAGINATION/export_imagined.py" --rollouts "$RUN/imagined/all" --task-id "$t" \
      --keep "$KEEP" --out "$RUN/lerobot_imagined/task_$t" \
      > "$RUN/lerobot_imagined.task_$t.log" 2>&1 &
    pids+=($!)
  done
  for t in "${pids[@]}"; do wait "$t" || return 1; done
}

sft() {  # sft demos|imagined
  local arm=$1 paths=$DEMOS ept=$DEMOS_PER_TASK t datasets=() w weights data
  data=("data.train_data_paths=$DEMOS" "data.episodes_per_task=$DEMOS_PER_TASK")
  if [ "$arm" = imagined ]; then
    # Tasks without a kept episode get no dataset; the others share half the batch.
    for t in "$RUN"/lerobot_imagined/task_*; do
      [ "$($PY -c "import json; print(json.load(open('$t/selection.json'))['episodes'])")" -gt 0 ] \
        && datasets+=("$t")
    done
    w=$($PY -c "print(0.5 / ${#datasets[@]})")
    weights=0.5
    for t in "${datasets[@]}"; do paths+=",$t"; ept+=",null"; weights+=",$w"; done
    data=("data.train_data_paths=[$paths]" "data.episodes_per_task=[$ept]"
      "data.sampling_weights=[$weights]")
  fi
  mkdir -p "$RUN/sft/$arm"
  echo "${data[*]}" > "$RUN/sft/$arm.log"
  EMBODIED_PATH=$REPO/examples/sft REPO_PATH=$REPO $PY examples/sft/train_vla_sft.py \
    --config-path "$REPO/examples/sft/config/" --config-name libero_spatial_sft_multitask_dit \
    runner.logger.log_path="$RUN/sft/$arm" runner.logger.experiment_name="$arm" \
    runner.max_steps="$SFT_STEPS" runner.save_interval="$SFT_STEPS" \
    actor.micro_batch_size=64 actor.global_batch_size=256 \
    actor.model.model_path="$BASE_CKPT" \
    actor.optim.lr=1.0e-4 actor.optim.lr_warmup_steps=500 \
    actor.optim.total_training_steps="$SFT_STEPS" \
    "${data[@]}" data.cache_dir="$CACHE_DIR" \
    >> "$RUN/sft/$arm.log" 2>&1
}

sft_ckpt() {
  echo "$RUN/sft/$1/$1/checkpoints/global_step_$SFT_STEPS/actor/model_state_dict/full_weights.pt"
}

test_eval() {  # test_eval <name> <ckpt>
  mkdir -p "$RUN/eval"
  collect "$2" "$RUN/eval/$1" --reruns "$TEST_RERUNS" ${TEST_TRIALS:+--trials "$TEST_TRIALS"}
}

monitor() {
  local m=$RUN/monitor
  mkdir -p "$m/real"
  ln -sfn "$VAL_REAL/episodes" "$m/real/episodes"
  cp "$VAL_REAL/summary.json" "$m/real/summary.json"
  "$OPENSORA_VENV/bin/python" "$IMAGINATION/imagine_rollouts.py" --mode replay \
    --real "$m/real" --wm-ckpt "$WM_CKPT" --out "$m/replay" \
    --reruns "$($PY -c "import json; print(json.load(open('$m/real/summary.json'))['reruns'])")" \
    --anchor-step "$ANCHOR_STEP" --num-envs 10 --seed "$SEED" --no-fused-kernels \
    > "$m/replay.log" 2>&1 || return 1
  judge "$m/real" && judge "$m/replay" || return 1
  $PY "$IMAGINATION/wm_fidelity.py" --real "$m/real" --replay "$m/replay" \
    --out "$m/fidelity.json" > "$m/fidelity.log" 2>&1
}

report() {
  local extra=()
  [ -f "$RUN/monitor/fidelity.json" ] && extra=(--fidelity "$RUN/monitor/fidelity.json")
  $PY "$IMAGINATION/report.py" --out "$RUN/report" \
    --arms "Base=$RUN/eval/base" "Demos only=$RUN/eval/demos" "Demos + imagined=$RUN/eval/imagined" \
    --selections "$RUN"/lerobot_imagined/task_* "${extra[@]}" > "$RUN/report.log" 2>&1
}

stage real real
[ -n "$ANCHOR_METHOD" ] && stage anchors select_anchors
stage imagine imagine
stage judge judge "$RUN/imagined/all"
stage export export_kept
stage sft_demos sft demos
stage sft_imagined sft imagined
if [ -n "$BASE_EVAL" ]; then
  mkdir -p "$RUN/eval" && ln -sfn "$(realpath "$BASE_EVAL")" "$RUN/eval/base"
else
  stage eval_base test_eval base "$BASE_CKPT"
fi
stage eval_demos test_eval demos "$(sft_ckpt demos)"
stage eval_imagined test_eval imagined "$(sft_ckpt imagined)"
[ "$MONITOR" = 1 ] && stage monitor monitor
stage report report
log "protocol finished: $RUN/report.md"
