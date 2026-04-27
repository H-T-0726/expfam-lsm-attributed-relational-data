"""Scenario A launcher: True X=Poisson, Y=Bernoulli"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reproduction" / "src"))

from exp_scenario_lib import run_scenario

if __name__ == "__main__":
    result = run_scenario("poisson", "bernoulli", "A", seed_offset=0)
    print("\n=== Scenario A Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
