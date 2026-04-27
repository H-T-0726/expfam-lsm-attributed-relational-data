"""Scenario C launcher: True X=Bernoulli, Y=Gaussian"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reproduction" / "src"))

from exp_scenario_lib import run_scenario

if __name__ == "__main__":
    result = run_scenario("bernoulli", "gaussian", "C", seed_offset=2000)
    print("\n=== Scenario C Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
