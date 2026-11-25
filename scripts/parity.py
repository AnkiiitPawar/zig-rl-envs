"""Bit-for-bit parity with Gymnasium's classic-control environments.

For each environment, sequences of actions are stepped by Gymnasium's own
environment classes (gymnasium.envs.classic_control, without wrappers) and by
`zrl replay`, from the same starting states:

  episodes     from Gymnasium's seeded resets, with random actions, and for
               CartPole also with a simple balancing controller that keeps
               the pole up for hundreds of steps;
  wide states  starting states drawn across everything the dynamics can
               reach, five steps each.

Every observation (float32), reward (float64) and termination flag must be
identical. The script first checks that NumPy's float64 sin and cos return
the C library's bits, which is what the parity rests on.

    ZIG="python3 -m ziglang" python3 scripts/parity.py [--quick]   # docs/data/parity.json
"""
import argparse
import math
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.envs.classic_control import AcrobotEnv, CartPoleEnv, PendulumEnv

from common import save_json, zrl

warnings.filterwarnings("ignore")
PI = math.pi
ENVS = {
    "cartpole": dict(cls=CartPoleEnv, obs_dim=4, discrete=2, wide=[(-2.4, 2.4), (-3, 3), (-0.21, 0.21), (-3.5, 3.5)]),
    "pendulum": dict(cls=PendulumEnv, obs_dim=3, discrete=None, wide=[(-20, 20), (-8, 8)]),
    "acrobot": dict(cls=AcrobotEnv, obs_dim=6, discrete=3, wide=[(-PI, PI), (-PI, PI), (-4 * PI, 4 * PI), (-9 * PI, 9 * PI)]),
}


def libm_check(n=200_000):
    x = np.random.default_rng(0).uniform(-50, 50, n)
    return int((np.sin(x) != np.array([math.sin(v) for v in x])).sum() + (np.cos(x) != np.array([math.cos(v) for v in x])).sum())


def balance(obs):
    return int(obs[2] + 0.5 * obs[3] > 0)


def run_gym(name, starts, policies, steps):
    """Steps Gymnasium from each start. A policy is a list of actions, or a
    function of the observation (whose choices are then recorded)."""
    spec = ENVS[name]
    env = spec["cls"]()
    env.reset(seed=0)
    actions, records = [], []
    for s0, pol in zip(starts, policies):
        env.unwrapped.state = s0.copy() if name != "acrobot" else s0.astype(np.float64)
        if name == "cartpole":
            env.unwrapped.steps_beyond_terminated = None
        obs = None
        acts, rec = [], []
        for t in range(steps):
            if callable(pol):
                a = pol(np.asarray(env.unwrapped.state)) if obs is None else pol(obs)
            else:
                a = pol[t]
            acts.append(a)
            if spec["discrete"]:
                obs, r, term, _, _ = env.step(int(a))
            else:
                obs, r, term, _, _ = env.step(np.array([a], dtype=np.float32))
            rec.append((np.asarray(obs, dtype=np.float32).tobytes(), struct.pack("<d", float(r)), bool(term)))
            if term:
                break
        acts += [0] * (steps - len(acts))
        actions.append(acts)
        records.append(rec)
    return actions, records


def run_zrl(exe, name, starts, actions, steps, tmp):
    spec = ENVS[name]
    inp = tmp / f"{name}.in"
    out = tmp / f"{name}.out"
    with open(inp, "wb") as fh:
        fh.write(struct.pack("<II", len(starts), steps))
        for s0, acts in zip(starts, actions):
            fh.write(np.asarray(s0, dtype=np.float64).astype("<f8").tobytes())
            if spec["discrete"]:
                fh.write(np.asarray(acts, dtype=np.uint8).tobytes())
            else:
                fh.write(np.asarray(acts, dtype="<f4").tobytes())
    subprocess.run([exe, "replay", "--env", name, "--in", str(inp), "--out", str(out)], check=True)
    data = out.read_bytes()
    rec = spec["obs_dim"] * 4 + 9
    res = []
    for k in range(len(starts)):
        seq = []
        for t in range(steps):
            off = (k * steps + t) * rec
            b = data[off:off + rec]
            seq.append((b[:spec["obs_dim"] * 4], b[spec["obs_dim"] * 4:spec["obs_dim"] * 4 + 8], b[-1] == 1))
        res.append(seq)
    return res


def compare(gym_recs, zrl_recs):
    steps = mismatches = 0
    first = None
    for k, (g, z) in enumerate(zip(gym_recs, zrl_recs)):
        for t, rec in enumerate(g):
            steps += 1
            if rec != z[t]:
                mismatches += 1
                if first is None:
                    first = (k, t)
    return steps, mismatches, first


def suite(name, quick):
    spec = ENVS[name]
    rng = np.random.default_rng({"cartpole": 1, "pendulum": 2, "acrobot": 3}[name])
    env = spec["cls"]()
    cases = []
    # Episodes from Gymnasium's resets.
    n_ep = {"cartpole": 2000, "pendulum": 200, "acrobot": 200}[name] // (10 if quick else 1)
    steps = {"cartpole": 500, "pendulum": 200, "acrobot": 500}[name]
    starts = []
    for k in range(n_ep):
        env.reset(seed=10_000 + k)
        starts.append(np.array(env.unwrapped.state, dtype=np.float64))
    if spec["discrete"]:
        pols = [list(rng.integers(0, spec["discrete"], steps)) for _ in starts]
    else:
        pols = [list(rng.uniform(-3, 3, steps).astype(np.float32)) for _ in starts]
    cases.append(("episodes, random actions", starts, pols, steps))
    if name == "cartpole":
        cases.append(("episodes, balancing controller", starts[: n_ep // 4], [balance] * (n_ep // 4), steps))
    # Wide states, five steps each.
    n_wide = 20_000 // (10 if quick else 1)
    wide = [np.array([rng.uniform(lo, hi) for lo, hi in spec["wide"]]) for _ in range(n_wide)]
    if spec["discrete"]:
        pols = [list(rng.integers(0, spec["discrete"], 5)) for _ in wide]
    else:
        pols = [list(rng.uniform(-3, 3, 5).astype(np.float32)) for _ in wide]
    cases.append(("wide states, 5 steps", wide, pols, 5))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="a tenth of the sequences (what CI runs)")
    args = ap.parse_args()
    bad_trig = libm_check()
    print(f"NumPy sin/cos differing from the C library's on 400,000 calls: {bad_trig}")
    exe = zrl()
    ok = bad_trig == 0
    results = {"gymnasium": gymnasium.__version__, "numpy": np.__version__, "trig_mismatches": bad_trig, "suites": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ENVS:
            for label, starts, pols, steps in suite(name, args.quick):
                actions, gym_recs = run_gym(name, starts, pols, steps)
                zrl_recs = run_zrl(exe, name, starts, actions, steps, tmp)
                n, bad, first = compare(gym_recs, zrl_recs)
                terminated = sum(1 for g in gym_recs if g and g[-1][2])
                ok &= bad == 0
                print(f"{'ok  ' if bad == 0 else 'FAIL'} {name:9} {label:32} {len(starts):6} sequences {n:8} steps "
                      f"{terminated:5} terminations  mismatches {bad}" + (f" (first: sequence {first[0]}, step {first[1]})" if first else ""))
                results["suites"].append({"env": name, "case": label, "sequences": len(starts), "steps": n,
                                          "terminations": terminated, "mismatches": bad})
    if not args.quick:
        save_json("parity.json", results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
