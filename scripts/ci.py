"""The checks that run in CI, after the unit tests.

  1. Parity with Gymnasium, on a tenth of the sequences of scripts/parity.py:
     every observation, reward and termination flag must be bit-identical.
  2. PPO on CartPole, built with safety checks on (ReleaseSafe), seed 1, the
     default 500,000 steps: the mean return of the last 100 training episodes
     must reach 475, the final policy must average at least 475 over 100 new
     episodes, and a second run with the same seed must give the same result.

    ZIG="python3 -m ziglang" python3 scripts/ci.py
"""
import json
import subprocess
import sys

from common import ROOT, zrl


def main():
    ok = subprocess.run([sys.executable, str(ROOT / "scripts" / "parity.py"), "--quick"]).returncode == 0
    exe = zrl("ReleaseSafe")
    runs = []
    for _ in range(2):
        out = subprocess.run([exe, "train", "--env", "cartpole", "--seed", "1"], check=True, capture_output=True,
                             text=True).stdout
        runs.append(json.loads(out))
    a, b = runs
    same = {k: v for k, v in a.items() if "seconds" not in k} == {k: v for k, v in b.items() if "seconds" not in k}
    solved = a["solved_step"] is not None and a["eval_mean_return"] >= 475
    print(f"{'ok  ' if solved else 'FAIL'} PPO on CartPole: solved at step {a['solved_step']}, final policy "
          f"{a['eval_mean_return']:.1f} over 100 episodes")
    print(f"{'ok  ' if same else 'FAIL'} the same seed gives the same run")
    ok &= solved and same
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
