# checkpoint_states.npz — 保管場所

このディレクトリの `checkpoint_states.npz`（49 MB）は repository 外に保管している。
このリポジトリは `.npz` を追跡していないため、48 MB を git 履歴へ永久に載せない判断をした。

- 保管先: `D:\tento\kennkyu_artifacts\em_convergence_sensitivity_20260905\checkpoint_states.npz`
- サイズ: 50,922,968 bytes
- sha256: `5e4390f1c61edba63acd3b5e055816331fd63758a46642b200b923e93d40af84`

## 中身

fit × checkpoint ごとに `<state_key>_Z`, `_Zsamples`, `_F`, `_sigma`,
`_scalars`(w0, w, var_z)。2,688 checkpoint 分。
S1 / S3 / Q_strict / S2 を独立に再計算するための生の状態であり、
再生成には 448 fit・約 6 時間の再実行が必要。

## 再監査するとき

`audit_em_convergence_sensitivity.py` はこのファイルの存在を要求する。
再実行する場合は上記の保管先からこのディレクトリへ戻すこと。

**監査は既に実行済みで、その結果は `audit_report.json` にある**
（verdict PASS、2,688 checkpoint すべてを再計算、BLOCKER 0 / HIGH 0 / MEDIUM 0）。
