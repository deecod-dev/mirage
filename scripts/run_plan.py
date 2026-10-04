"""Run the whole v2 plan across every available GPU, then analyse the result.

One command, start to finish. Scheduling rules:

* Each experiment has a **leader** seed that generates the shared clean traces and
  **followers** that only inject. A follower cannot start until its leader has
  finished, or the two would race to generate the same clean trace.
* Leaders of *different* experiments have nothing in common, so they all start at once.
* Workers are pinned one per GPU and pull the longest eligible job first, so the tail
  of the run stays balanced instead of leaving a card idle behind one slow seed.

Everything is resumable: traces are content-addressed, so re-running this script after
an interruption picks up exactly where it stopped and recomputes nothing.

    python scripts/run_plan.py                 # all GPUs, full plan
    python scripts/run_plan.py --gpus 0        # one GPU
    python scripts/run_plan.py --dry-run       # print the schedule and exit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

#: (config stem, seeds, approximate traces per seed) — cost is used only for ordering.
PLAN: list[tuple[str, list[int], int]] = [
    ("E1_pipeline", [11], 200),
    ("E3_positive_control", [11, 23, 37], 1800),
    ("E4_fingerprint", [11, 23, 37, 53, 71], 4400),
]

#: A two-minute rehearsal of the same machinery: two seeds of the injector pre-flight,
#: which exercises leader/follower gating, the shared clean cache and both GPUs.
SMOKE_PLAN: list[tuple[str, list[int], int]] = [("P2_injector_check", [11, 23], 312)]

ANALYSES: list[tuple[str, list[str]]] = [
    ("fingerprint", ["scripts/analyze_fingerprint.py",
                     "--train-experiment", "E3_positive_control_v2",
                     "--test-experiment", "E4_fingerprint_v2",
                     "--output", "results/E4_fingerprint_v2.json"]),
    ("transfer", ["scripts/analyze_transfer.py",
                  "--train-experiment", "E3_positive_control_v2",
                  "--test-experiment", "E4_fingerprint_v2",
                  "--output", "results/E5_transfer_v2.json"]),
    ("validity audit", ["scripts/audit_probe_validity.py",
                        "--train-experiment", "E3_positive_control_v2",
                        "--test-experiment", "E4_fingerprint_v2",
                        "--output", "results/E4_validity_audit_v2.json"]),
]


@dataclass
class Job:
    config: str
    seed: int
    cost: int
    leader: bool
    status: str = "pending"
    seconds: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.config}:seed{self.seed}"


@dataclass
class Experiment:
    config: str
    leader: Job
    followers: list[Job] = field(default_factory=list)
    leader_done: threading.Event = field(default_factory=threading.Event)


def detect_gpus() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
        return [int(line) for line in out.split() if line.strip().isdigit()]
    except Exception:
        return [0]


def run_one(job: Job, gpu: int) -> None:
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{job.config}_seed{job.seed}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false"}
    started = time.monotonic()
    with log.open("w") as handle:
        result = subprocess.run(
            [sys.executable, "scripts/run_server_experiment.py",
             "--config", f"configs/experiments/{job.config}.yaml", "--seed", str(job.seed)],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    job.seconds = time.monotonic() - started
    job.status = "done" if result.returncode == 0 else "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU indices to use (default: every GPU nvidia-smi reports)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="rehearse the scheduler on the pre-flight config (~3 min) "
                             "instead of running the full plan")
    args = parser.parse_args()

    gpus = args.gpus if args.gpus is not None else detect_gpus()
    plan = SMOKE_PLAN if args.smoke else PLAN
    experiments: list[Experiment] = []
    for config, seeds, cost in plan:
        leader = Job(config, seeds[0], cost, leader=True)
        followers = [Job(config, s, cost, leader=False) for s in seeds[1:]]
        experiments.append(Experiment(config, leader, followers))
    jobs = [e.leader for e in experiments] + [f for e in experiments for f in e.followers]

    print(f"GPUs: {gpus}   jobs: {len(jobs)}   "
          f"traces: ~{sum(j.cost for j in jobs):,}")
    for e in experiments:
        print(f"  {e.config:24s} leader seed {e.leader.seed}"
              + (f", followers {[f.seed for f in e.followers]}" if e.followers else ""))
    if args.dry_run:
        return 0

    ordered = sorted(jobs, key=lambda j: (-j.cost, j.seed))
    by_config = {e.config: e for e in experiments}
    lock = threading.Lock()
    failures: list[Job] = []
    overall = time.monotonic()

    def claim() -> Job | None:
        """Longest eligible job, or None. A follower is eligible once its leader is
        done; if the leader failed, its followers are dropped rather than left to
        spin forever waiting for clean traces that will never exist."""
        with lock:
            for job in ordered:
                if job.status != "pending":
                    continue
                if job.leader or by_config[job.config].leader_done.is_set():
                    job.status = "running"
                    return job
                if by_config[job.config].leader.status == "failed":
                    job.status = "skipped"
            return None

    def anything_running() -> bool:
        with lock:
            return any(j.status == "running" for j in ordered)

    def worker(gpu: int) -> None:
        while True:
            job = claim()
            if job is None:
                # Nothing eligible. Either a leader is still building clean traces —
                # wait for it — or the plan is finished.
                if not anything_running():
                    return
                time.sleep(5)
                continue
            with lock:
                print(f"[gpu{gpu}] start {job.name}", flush=True)
            run_one(job, gpu)
            if job.leader:
                by_config[job.config].leader_done.set()
            with lock:
                mark = "ok" if job.status == "done" else "FAILED"
                print(f"[gpu{gpu}] {mark:6s} {job.name} in {job.seconds / 60:.1f} min", flush=True)
                if job.status == "failed":
                    failures.append(job)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed = (time.monotonic() - overall) / 3600
    print(f"\ngeneration finished in {elapsed:.2f} h")
    for job in jobs:
        print(f"  {job.status:7s} {job.name:34s} {job.seconds/60:7.1f} min")
    if failures:
        print(f"\n{len(failures)} job(s) FAILED — see logs/. Re-run this script; it resumes.")
        return 1

    if args.skip_analysis or args.smoke:
        return 0
    print("\nrunning analyses")
    for label, command in ANALYSES:
        print(f"\n=== {label} ===", flush=True)
        if subprocess.run([sys.executable, *command], cwd=ROOT).returncode != 0:
            print(f"analysis '{label}' failed")
            return 1

    gate = json.loads((ROOT / "results/E4_fingerprint_v2.json").read_text())
    arm = gate["arms"][gate["gate_arm"]]["overall"]
    print(f"\nG-4 ({gate['gate_arm']} arm): {arm['point']:.4f} "
          f"[{arm['ci_low']:.4f}, {arm['ci_high']:.4f}] -> "
          f"{'PASS' if gate['gate_g4'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
