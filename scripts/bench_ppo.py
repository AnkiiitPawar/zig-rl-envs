"""Train PPO with zrl on each environment over several seeds.

Each run uses the environment's default settings (see src/ppo.zig), one
thread, and its full step budget. It records when the mean return of the last
100 training episodes first reached the target (CartPole 475, Acrobot -100,
Pendulum -200), the mean training return at the end, and the mean return of
the final deterministic policy over 100 new episodes. Runs go two at a time on
the 2-vCPU machine.

    ZIG="python3 -m ziglang" python3 scripts/bench_ppo.py [--seeds 10] [--jobs 2]
    # docs/data/bench_ppo.json; each run's result and learning curve in .cache/ppo/

Runs whose result is already in .cache/ppo/ are not repeated (use --force).
"""
import argparse
import concurrent.futures as cf
import json
import subprocess

from common import CACHE, save_json, zrl

ENVS = ["cartpole", "acrobot", "pendulum"]


def run(job):
    exe, env, seed, force = job
    log = CACHE / "ppo" / f"{env}-{seed}.csv"
    res = CACHE / "ppo" / f"{env}-{seed}.json"
    if force or not res.exists():
        subprocess.run([exe, "train", "--env", env, "--seed", str(seed), "--log", str(log), "--json", str(res)],
                       check=True, capture_output=True)
    return json.loads(res.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    exe = zrl()
    (CACHE / "ppo").mkdir(parents=True, exist_ok=True)
    jobs = [(exe, env, s, args.force) for env in args.envs.split(",") for s in range(1, args.seeds + 1)]
    rows = []
    with cf.ProcessPoolExecutor(args.jobs) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    save_json("bench_ppo.json", {"seeds": args.seeds, "jobs": args.jobs, "runs": rows})


if __name__ == "__main__":
    main()
