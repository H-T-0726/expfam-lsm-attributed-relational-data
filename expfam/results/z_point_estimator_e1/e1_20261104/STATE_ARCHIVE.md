# 状態ファイルの保管場所（Issue #141 E1）

`.npz` はリポジトリで追跡しないため、状態ファイルは repository 外に置く。

| stage | 保管先 | size | sha256 |
|---|---|---|---|
| smoke | `D:\tento\kennkyu_artifacts\z_point_estimator_e1_20261104\smoke\states.npz` | 27,969 bytes | `3a336bba606a9cc10fb4f4f4024e7a2262796dd46e629a438d0dbad67ec198f3` |
| full | `D:\tento\kennkyu_artifacts\z_point_estimator_e1_20261104\full\states.npz` | 2,238,319 bytes | `393a6d31af60a51ddf691716632650c1f72538f7da3dce15c7b544b860df9cd2` |

- 中身: 各 fit（`<part>_<replicate>_<regime>_<arm>`）について次を保存した。
  - `Zsamples`（最後の E-step の 5 サンプル）、`F`、`sigma`、`scalars`（w0, w, var_z）
  - `Zmode`（Z0 = last からの joint mode）、`Zmode_from_mean5`（Z0 = mean5 からの joint mode）
- full の実行ログ: `D:\tento\kennkyu_artifacts\e1_full_console.log`（sha256 `2ed23c2b156c2fee76c2b302c35fa6cdd5e19f0c5103da11915ce75da4fc6921`）
- full の sha256 は `full/runinfo.json` の `states_sha256` にも記録されている。
