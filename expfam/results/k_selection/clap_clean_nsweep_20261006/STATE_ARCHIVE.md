# 状態ファイルの保管場所（Issue #128 実験 A1）

`.npz` はリポジトリで追跡しないため、状態ファイルは repository 外に置く。

| stage | 保管先 | sha256 |
|---|---|---|
| pilot | `D:\tento\kennkyu_artifacts\clap_clean_nsweep_20261006\pilot\states.npz` | `e0b5df70a383fd9e11c2b27be95e330b88fe69dab249a24bcf30506bdd18c49e` |
| full | `D:\tento\kennkyu_artifacts\clap_clean_nsweep_20261006\full\states.npz`（9,813,678 bytes） | `ec3bbc5ed9b60d8cfcb5fe78e8a8378e944a3b366c177005b5e7d76c50dc380f` |

中身: 全 fit の joint mode `Ẑ`（`<tag>_Zhat`）と、再現した K_TRUE=1,3 の fit の
`<tag>_Zsamples`, `<tag>_F`, `<tag>_scalars`(w0, w, var_z)。
K_TRUE=5 の θ̂ と E-step サンプルは `em_convergence_sensitivity_20260905/checkpoint_states.npz`
（sha256 `5e4390f1…af84`）の checkpoint 8 にある。

`audit_clap_clean_nsweep.py` は `runinfo.json` の `states_file` と sha256 を照合する。
