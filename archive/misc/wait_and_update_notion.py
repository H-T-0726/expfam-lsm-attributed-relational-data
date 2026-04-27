"""
Exp2 完了を待機して notion_v6_final.py を再実行するスクリプト
"""
import time, os, subprocess
from pathlib import Path

RES = Path("C:/kennkyu/expfam/results")
required = [RES / "exp2_bic_A.csv", RES / "exp2_bic_B.csv", RES / "exp2_bic_C.csv"]

print("Exp2 完了を待機中...", flush=True)
while True:
    done = all(f.exists() for f in required)
    if done:
        print("\n全 exp2_bic_*.csv が揃いました！Notion を最終更新します...", flush=True)
        result = subprocess.run(
            ["python", "notion_v6_final.py"],
            cwd="C:/kennkyu",
            capture_output=False,
        )
        print(f"更新完了（returncode={result.returncode}）")
        break
    missing = [f.name for f in required if not f.exists()]
    print(f"  待機中... 未完了: {missing}", flush=True)
    time.sleep(300)  # 5分ごとに確認
