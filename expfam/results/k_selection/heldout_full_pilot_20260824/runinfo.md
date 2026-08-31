# Phase 7e full held-out K-selection pilot — runinfo

- issue: #43
- branch: `experiment/full-heldout-k-selection-pilot`
- RUN_CODE_SHA: `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a`
- base main SHA: `a11406ca5e93c216bd4faa875fdbe0ca73c406c6`
- start (UTC): 2026-08-23T18:23:52.927690+00:00
- start (local): 2026-08-24T03:23:52.927702+09:00
- finish (UTC): 2026-08-23T18:28:59.333196+00:00
- command: `python tools/research_audit/run_heldout_k_selection_pilot.py --full --allow-em --confirm-full-pilot`
- Python: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
- NumPy: 2.3.5
- platform: Windows-11-10.0.26200-SP0
- candidate K: [1, 2, 3, 4, 5, 6, 7]
- starts: [1, 2]
- replicates: [1, 2, 3]
- expected fit count: 42
- actual fit count: 42
- targets created: 3
- score rows: 42
- failure state: none
- stdout log: `expfam/results/k_selection/heldout_full_pilot_20260824/stdout.log`

## git status before

```
(clean)
```

## git status after scientific execution

```
?? expfam/results/k_selection/heldout_full_pilot_20260824/
```

## generated artifacts

- `aggregate_summary.csv`
- `fit_results.csv`
- `manifest.csv`
- `replicate_selection.csv`
- `score_by_k.csv`
- `runinfo.json`
- `runinfo.md`
- `stdout.log`

## per-replicate provenance

### replicate 1

- data_seed: `41001`
- split_seed: `42001`
- x_hash: `7b70d89f1ee1f99cbbef3d882fe03b234937f2e87b6c566fb1df21d350f6a43c`
- training_y_hash: `9d1ee824349ca4631da6f58b8f53b0fd2da9049f4261b844207fa0a627e2dd98`
- train_mask_hash: `2ed450a0e79277ccd6dda97bafa504854a564af0cb41a3cd02c7b0332341dbd5`
- test_mask_hash: `387bda3b61226f2e22e1ca9065e35213fccc999c0b4abfa89dcce5562c2093ce`
- fit_provenance_hash: `ba24bc13536db54d1f69f6e462146e4e16a1d15d46bf4d5692b09c8356acc3ea`
- target_topology_hash: `a826f7ab064bb4c2c142de42c3b82df047b7eec682b5a711dcd9408fde874645`
- preprocessing_hash: `8c050737bf3c32619cc617a01e350826b4f343c196c484ae53f2f6bd2054b48b`
- score_config_hash: `aae47803a349add9823cd36ed45cc9c2d81bbee435af4c9d64cbb6359de0fa8b`
- score_target_hash: `['150e6c43d371fc87ef05294f3c7bfea88acc822ef7ab701a91bb6f0bb5b7d68b']`

### replicate 2

- data_seed: `41002`
- split_seed: `42002`
- x_hash: `f9838a7e47c44c2d0c29b5476c9da49566463bd3e5cb08b78d8787041f2fde31`
- training_y_hash: `109009099f797df5942c129b4f5541c2afb454299f0e35873f054a947bf3100e`
- train_mask_hash: `ed37ddf1607a13084d9d4abcb81153324c0ad6022561dd00e240e14f4b85b96a`
- test_mask_hash: `cb94a1861b00649eb0a7829154922ce5351407adcb7efa36b7f8d7aaf0b75578`
- fit_provenance_hash: `8fdb97e69b6333e29cc7221bcfa970343a7675bad3e2cd7795af871b8a0f4402`
- target_topology_hash: `2f6d3501cf95ff79f4cabee669be6d2c6852324112f3b6c40c4aab9db4f048c2`
- preprocessing_hash: `8c050737bf3c32619cc617a01e350826b4f343c196c484ae53f2f6bd2054b48b`
- score_config_hash: `aae47803a349add9823cd36ed45cc9c2d81bbee435af4c9d64cbb6359de0fa8b`
- score_target_hash: `['eda298e315fa6fea26371df36a9f7c6d595d61963510ff91d521f452da34957d']`

### replicate 3

- data_seed: `41003`
- split_seed: `42003`
- x_hash: `565ba209d1f798a277291bd8f95149100760cbcbf3fc61f2a9b3341d261c7244`
- training_y_hash: `59dbe0e45b531af983af7d8af3592c4f0b28a7c362356fef8e88d8a99345fce3`
- train_mask_hash: `dd38a6dc0c729eb26b133f73a06b99d12c5088e5248da46847456f733bcb326f`
- test_mask_hash: `3e56e119afbb8aabbd44551af3351ee6a55c0b96c7d9e6aac153faecff6a9110`
- fit_provenance_hash: `e8ea7182198b9f45ab36c6c84007feefcb8013f4396972db65ce79a077e10c19`
- target_topology_hash: `067e35129905890a6822bbf1a5906e042fc47e1ef39c60a363827de0f2989e09`
- preprocessing_hash: `8c050737bf3c32619cc617a01e350826b4f343c196c484ae53f2f6bd2054b48b`
- score_config_hash: `aae47803a349add9823cd36ed45cc9c2d81bbee435af4c9d64cbb6359de0fa8b`
- score_target_hash: `['77bb6de605581d8f3f828776959a726332cdc2168cb64a08f12f2c2897087eb8']`

