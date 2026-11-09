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

"""Sample a frozen pool of new LIBERO initial states and split it.

Each task's BDDL placement sampler draws states the same way LIBERO drew its
benchmark ``init_files``, with robosuite's default Gaussian joint noise
(0.02 rad) on the robot's start pose; LIBERO's env wrapper turns that noise off
unless asked. A sampled state is kept only if it passes the checks
the benchmark states pass, measured over the same zero-action wait the env runs
after ``set_init_state`` (``num_steps_wait``):

- the goal is not already satisfied after the wait;
- no object moves farther horizontally during the wait than that object does
  in any benchmark state of the task, or ``--xy-tolerance`` if larger
  (objects start above their resting spot and drop, so vertical motion is
  expected; sliding or toppling is not);
- its settled object layout is no closer to a benchmark (test) state than the
  two closest benchmark states are to each other. Two layouts are as far apart
  as the object that differs most in horizontal position. Fixtures (cabinet,
  stove) are left out: ``reset`` re-places them in the model and the state
  does not store them, so they vary by a few cm whatever state is loaded.

The kept pool is split per task as ``validation = pool[0:V]``,
``imagination = pool[V:V+I]``, and the rest is reserve, so growing a split
later keeps its existing states. Each split directory holds one
``<task.init_states_file>`` per task in LIBERO's format and is loaded with the
env config key ``init_states_dir``. Test states are the benchmark's own
``init_files`` (``init_states_dir: null``). ``split.json`` records the settings,
the per-task pool indices of each split, and the filter statistics.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--pool", type=int, default=100, help="Kept states per task.")
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--imagination", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=15)
    parser.add_argument(
        "--max-draws",
        type=int,
        default=1000,
        help="Sampler draws per task before giving up on filling the pool.",
    )
    parser.add_argument(
        "--xy-tolerance",
        type=float,
        default=0.001,
        help="Smallest horizontal drift in meters treated as unstable.",
    )
    return parser.parse_args()


def settle(env, state: np.ndarray, num_steps: int):
    """Set ``state``, run the env's zero-action wait, and measure it.

    Returns the settled horizontal position of each movable object, each
    object's horizontal displacement during the wait in meters, and whether
    the goal holds afterwards.
    """
    env.set_init_state(state)
    inner = env.env
    body_ids = [inner.obj_body_id[name] for name in inner.objects_dict]
    start = inner.sim.data.body_xpos[body_ids, :2].copy()
    action = np.zeros(7)
    action[-1] = -1.0
    for _ in range(num_steps):
        env.step(action)
    end = inner.sim.data.body_xpos[body_ids, :2].copy()
    return end, np.linalg.norm(end - start, axis=1), bool(env.check_success())


def layout_distance(layout: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Distance from ``layout`` to each reference: the largest per-object gap."""
    return np.linalg.norm(refs - layout[None], axis=2).max(axis=1)


def sample_task(args, bench, task_id: int) -> tuple[np.ndarray, dict]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = bench.get_task(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ),
        camera_heights=64,
        camera_widths=64,
        initialization_noise="default",
    )
    task_seed = args.seed * 1000 + task_id
    env.seed(task_seed)
    np.random.seed(task_seed)

    bench_layouts, bench_moved = [], []
    for state in bench.get_task_init_states(task_id):
        env.reset()
        layout, moved, success = settle(env, state, args.num_steps_wait)
        if success:
            raise RuntimeError(
                f"Expected benchmark state of task {task_id} to start unsolved, "
                "found the goal satisfied after the wait."
            )
        bench_layouts.append(layout)
        bench_moved.append(moved)
    bench_layouts = np.stack(bench_layouts)
    max_moved = np.maximum(np.max(bench_moved, axis=0), args.xy_tolerance)
    object_names = list(env.env.objects_dict)
    gaps = np.array(
        [
            layout_distance(bench_layouts[i], np.delete(bench_layouts, i, axis=0)).min()
            for i in range(len(bench_layouts))
        ]
    )
    min_gap = float(gaps.min())

    def joint_offset_std(states) -> float:
        offsets = []
        for state in states:
            env.set_init_state(state)
            offsets.append(env.env.sim.data.qpos[:7] - env.env.robots[0].init_qpos)
        return float(np.std(offsets))

    kept, kept_moved, kept_gap = [], [], []
    rejected = {"solved": 0, "unstable": 0, "near_test": 0}
    draws = 0
    while len(kept) < args.pool:
        if draws == args.max_draws:
            raise RuntimeError(
                f"Expected {args.pool} states for task {task_id} within "
                f"{args.max_draws} draws, kept {len(kept)} (rejected {rejected})."
            )
        draws += 1
        env.reset()
        state = env.get_sim_state().copy()
        layout, moved, success = settle(env, state, args.num_steps_wait)
        if success:
            rejected["solved"] += 1
            continue
        if (moved > max_moved).any():
            rejected["unstable"] += 1
            continue
        gap = float(layout_distance(layout, bench_layouts).min())
        if gap < min_gap:
            rejected["near_test"] += 1
            continue
        kept.append(state)
        kept_moved.append(moved.max())
        kept_gap.append(gap)
    bench_noise = joint_offset_std(bench.get_task_init_states(task_id))
    pool_noise = joint_offset_std(kept)
    env.close()

    stats = {
        "task": task.language,
        "init_states_file": task.init_states_file,
        "draws": draws,
        "rejected": rejected,
        "max_xy_moved_m": dict(zip(object_names, max_moved.round(5).tolist())),
        "benchmark_min_gap_m": min_gap,
        "benchmark_median_gap_m": float(np.median(gaps)),
        "pool_max_xy_moved_m": float(max(kept_moved)),
        "pool_min_gap_to_test_m": float(min(kept_gap)),
        "pool_median_gap_to_test_m": float(np.median(kept_gap)),
        "benchmark_joint_noise_std_rad": bench_noise,
        "pool_joint_noise_std_rad": pool_noise,
    }
    return np.stack(kept), stats


def main() -> None:
    args = parse_args()
    if args.validation + args.imagination > args.pool:
        raise ValueError(
            f"Expected --validation + --imagination <= --pool, got "
            f"{args.validation} + {args.imagination} > {args.pool}."
        )
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    out = Path(args.out)
    splits = {
        "validation": (0, args.validation),
        "imagination": (args.validation, args.validation + args.imagination),
        "pool": (0, args.pool),
    }
    for name in splits:
        (out / name).mkdir(parents=True, exist_ok=True)

    # A finished task leaves stats/task_<id>.json, so a rerun with the same
    # settings resumes after it; the per-task seed makes the result identical.
    settings = {
        key: getattr(args, key)
        for key in (
            "task_suite",
            "pool",
            "validation",
            "imagination",
            "seed",
            "num_steps_wait",
            "xy_tolerance",
        )
    }
    (out / "stats").mkdir(exist_ok=True)
    tasks = []
    for task_id in range(bench.n_tasks):
        stats_path = out / "stats" / f"task_{task_id}.json"
        if stats_path.exists():
            done = json.loads(stats_path.read_text())
            if done["settings"] != settings:
                raise ValueError(
                    f"Expected {stats_path} written with settings {settings}, "
                    f"found {done['settings']}; use a new --out."
                )
            tasks.append(done["stats"])
            logger.info(f"task {task_id}: already sampled, skipping")
            continue
        states, stats = sample_task(args, bench, task_id)
        for name, (start, end) in splits.items():
            torch.save(states[start:end], out / name / stats["init_states_file"])
        stats_path.write_text(json.dumps({"settings": settings, "stats": stats}))
        tasks.append(stats)
        logger.info(
            f"task {task_id}: kept {args.pool} of {stats['draws']} draws, "
            f"rejected {stats['rejected']}, nearest test layout median "
            f"{stats['pool_median_gap_to_test_m']:.3f} m "
            f"(benchmark neighbors {stats['benchmark_median_gap_m']:.3f} m), "
            f"joint noise std {stats['pool_joint_noise_std_rad']:.4f} rad "
            f"(benchmark {stats['benchmark_joint_noise_std_rad']:.4f})"
        )

    split = {
        "task_suite": args.task_suite,
        "seed": args.seed,
        "num_steps_wait": args.num_steps_wait,
        "xy_tolerance_m": args.xy_tolerance,
        "test": {
            "init_states_dir": None,
            "states_per_task": [
                len(bench.get_task_init_states(i)) for i in range(bench.n_tasks)
            ],
        },
        "splits": {
            name: {"init_states_dir": str(out / name), "pool_index": [start, end]}
            for name, (start, end) in splits.items()
        },
        "tasks": tasks,
    }
    (out / "split.json").write_text(json.dumps(split, indent=1))
    logger.info(f"Wrote {out / 'split.json'}")


if __name__ == "__main__":
    main()
