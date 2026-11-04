# 状態ファイルの保管場所（Issue #128 実験 B）

`.npz` はリポジトリで追跡しないため、状態ファイルは repository 外に置く。

| stage | 保管先 | sha256 |
|---|---|---|
| pilot | `D:\tento\kennkyu_artifacts\paper_condition_k_criteria_20261006\pilot\states.npz`（799,509 bytes） | `c385901d09b7b46a35ef8efc17d84dc3aa83cab9be4f339f7e8f3fe63bcfbcf3` |
| full | `D:\tento\kennkyu_artifacts\paper_condition_k_criteria_20261006\full\states.npz`（19,989,709 bytes） | `db6a725668d6a55d010e2ea780aeaee2bf0dd11c65246abb80d0aacfee3c4005` |

中身: 各 fit（`K<K_TRUE>_r<rep>_k<K>_s<start>`）の `Zsamples`, `F`, `sigma`（対角）, `Zhat`（joint mode。Laplace 非 OK のときは停止点）, `scalars`（w0, w, var_z）。
実行ログ: `D:\tento\kennkyu_artifacts\b_pilot_console.log`, `D:\tento\kennkyu_artifacts\b_full_console.log`。

`audit_paper_condition_k_criteria.py` は `runinfo.json` の `states_file` と sha256 を照合する。
