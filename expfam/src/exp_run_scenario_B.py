"""Scenario B launcher: True X=Gaussian, Y=Poisson"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reproduction" / "src"))

from exp_scenario_lib import run_scenario

if __name__ == "__main__":
    result = run_scenario("gaussian", "poisson", "B", seed_offset=1000)
    print("\n=== Scenario B Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
