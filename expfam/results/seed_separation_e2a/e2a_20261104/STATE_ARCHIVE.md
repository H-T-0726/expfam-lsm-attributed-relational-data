# 状態ファイルの保管場所（Issue #142 E2a）

`.npz` はリポジトリで追跡しないため、状態ファイルは repository 外に置く。

| stage | 保管先 | size | sha256 |
|---|---|---|---|
| smoke | `D:\tento\kennkyu_artifacts\seed_separation_e2a_20261104\smoke\states.npz` | 64,029 bytes | `d95d25d656dcd9edcd3a3fe4745900c96ec7209c44768378b4b6b70d9ca0f60a` |
| full | `D:\tento\kennkyu_artifacts\seed_separation_e2a_20261104\full\states.npz` | 1,281,322 bytes | `84749cdb11a1e395b13c7af74e59b66c0a7f631211494127dde82ae71b05cd0d` |

- 中身: 各 fit（`<cell>_<replicate>_sparse_<arm>`）について次を保存した。
  - `Zsamples`、`Zest`、`F`、`sigma`、`scalars`（w0, w, var_z）
  - `Zmode`（Z0 = last）、`Zmode_from_mean5`
- 再利用した AA / BB は E1 の `D:\tento\kennkyu_artifacts\z_point_estimator_e1_20261104\full\states.npz`（sha256 `393a6d31…`、runner が読み込み時に照合する）。
- full の実行ログ: `D:\tento\kennkyu_artifacts\e2a_full_console.log`（sha256 `9be967763dc1317160c5279f3dd902756ffc7862156e7e811241c00853a3bead`）
