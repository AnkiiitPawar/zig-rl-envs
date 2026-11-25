"""Environment steps per second: zrl against Gymnasium and EnvPool.

Every implementation steps N copies of an environment with random actions,
resetting finished episodes, for at least a second, on one thread:

  zrl            the batched Zig environments (`zrl bench`);
  gymnasium      Gymnasium 1.4.0's environments in a Python loop (N = 1),
                 and its SyncVectorEnv;
  gym-numpy      Gymnasium's NumPy-vectorised CartPole (make_vec with
                 vectorization_mode="vector_entry_point"), CartPole only;
  envpool        EnvPool 1.2.8 (C++), synchronous, num_threads=1.

Actions are drawn before timing for zrl, and inside the loop (one NumPy call
per step) for the Python-driven ones.

    ZIG="python3 -m ziglang" python3 scripts/bench_envs.py   # docs/data/bench_envs.json
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import time
import types
import warnings

import gymnasium as gym
import numpy as np

from common import save_json, zrl

warnings.filterwarnings("ignore")
ENVS = {"cartpole": ("CartPole-v1", 2), "pendulum": ("Pendulum-v1", None), "acrobot": ("Acrobot-v1", 3)}
SIZES = [1, 16, 256, 4096]
MIN_SECONDS = 1.0


def import_envpool():
    """EnvPool imports every environment family when it is imported, and some
    of them need OpenGL, which this machine does not have. Only classic
    control is needed here, so the others are replaced by empty modules."""
    spec = importlib.util.find_spec("envpool")
    if spec is None:
        return None
    entry = (pathlib.Path(spec.origin).parent / "entry.py").read_text()
    for mod in re.findall(r"^import (envpool\.\S+)", entry, re.M):
        if mod.startswith("envpool.classic_control"):
            continue
        parts = mod.split(".")
        for k in range(2, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:k]), types.ModuleType(".".join(parts[:k])))
    import envpool
    return envpool


def actions(rng, n_actions, n):
    if n_actions is None:
        return rng.uniform(-2, 2, (n, 1)).astype(np.float32)
    return rng.integers(0, n_actions, n)


def timed(step, n):
    """Calls step() until MIN_SECONDS have passed; returns env steps per second."""
    calls = 0
    t0 = time.perf_counter()
    while True:
        for _ in range(10):
            step()
        calls += 10
        dt = time.perf_counter() - t0
        if dt >= MIN_SECONDS:
            return calls * n / dt


def bench_gym_loop(env_id, n_actions):
    env = gym.make(env_id)
    env.reset(seed=1)
    rng = np.random.default_rng(1)

    def step():
        a = actions(rng, n_actions, 1)[0]
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    return timed(step, 1)


def bench_gym_vector(env_id, n_actions, n, mode):
    env = gym.make_vec(env_id, num_envs=n, vectorization_mode=mode)
    env.reset(seed=1)
    rng = np.random.default_rng(1)

    def step():
        a = actions(rng, n_actions, n)
        env.step(a)
    return timed(step, n)


def bench_envpool(envpool, env_id, n_actions, n):
    env = envpool.make_gymnasium(env_id, num_envs=n, batch_size=n, num_threads=1, seed=1)
    env.reset()
    rng = np.random.default_rng(1)

    def step():
        a = actions(rng, n_actions, n)
        env.step(a if n_actions is not None else a)
    return timed(step, n)


def bench_zrl(exe, name, n):
    steps = max(200, int(4_000_000 / n))
    while True:
        out = subprocess.run([exe, "bench", "--env", name, "--envs", str(n), "--steps", str(steps)], check=True,
                             capture_output=True, text=True).stdout
        res = json.loads(out)
        if res["seconds"] >= MIN_SECONDS:
            return res["steps_per_second"]
        steps = int(steps * 1.5 * MIN_SECONDS / max(res["seconds"], 1e-3))


def main():
    exe = zrl()
    envpool = import_envpool()
    rows = []
    for name, (env_id, n_actions) in ENVS.items():
        for n in SIZES:
            impls = [("zrl", lambda: bench_zrl(exe, name, n))]
            if n == 1:
                impls.append(("gymnasium", lambda: bench_gym_loop(env_id, n_actions)))
            elif n <= 256:
                impls.append(("gymnasium", lambda: bench_gym_vector(env_id, n_actions, n, "sync")))
            if name == "cartpole" and n > 1:
                impls.append(("gym-numpy", lambda: bench_gym_vector(env_id, n_actions, n, "vector_entry_point")))
            if envpool is not None:
                impls.append(("envpool", lambda: bench_envpool(envpool, env_id, n_actions, n)))
            for impl, fn in impls:
                sps = fn()
                rows.append({"env": name, "envs": n, "impl": impl, "steps_per_second": sps})
                print(f"{name:9} {n:5} envs  {impl:10} {sps:14,.0f} steps/s", flush=True)
    save_json("bench_envs.json", {"gymnasium": gym.__version__, "envpool": getattr(envpool, "__version__", None),
                                  "min_seconds": MIN_SECONDS, "runs": rows})


if __name__ == "__main__":
    main()
