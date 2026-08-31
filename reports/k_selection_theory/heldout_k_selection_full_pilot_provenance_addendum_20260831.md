# Phase 7e full held-out K-selection pilot — provenance addendum

**日付:** 2026-08-31
**対象:** `expfam/results/k_selection/heldout_full_pilot_20260824/`
**対象 report:** `reports/k_selection_theory/heldout_k_selection_full_pilot_report_20260824.md`（**historical frozen record として不変**）
**Issue:** #43 / **PR:** #44
**RUN_CODE_SHA:** `b9311e64a7b36c0a8a9704fff0ee7b38efe36a8a`
**result commit:** `b816836f95945024f56ed7a4ac619e809bc16ded`

これは **append-only の forward correction** である。2026-08-24 の report・runinfo・stdout.log・
result CSV はいずれも書き換えていない。42 fits の再実行は行っていない。

---

## 1. なぜこの addendum があるか

PR #44 の independent Codex review が verdict `FIX_BEFORE_HUMAN_MERGE` を出し、
MEDIUM finding を 2 件指摘した。

| Finding | 内容 |
|---|---|
| **A** | `stdout.log` の生成経路（outer capture command）が repository 上に記録されておらず、保存 artifact だけからは独立に復元できない |
| **B** | artifact completeness / self-audit が header-only・missing-row 等で **fail-open** し得る |

科学結果についての review 判定は `selector arithmetic: PASS` / `leakage isolation: PASS` /
`statistical interpretation: PASS` / `model lineage: PASS` であり、
**selected K・arithmetic・seed・hash・42 saved fit rows のいずれも変更されていない。**

---

## 2. 保存されている inner Python command

`runinfo.json` の `command` フィールドに次が記録されている。

```
python tools/research_audit/run_heldout_k_selection_pilot.py --full --allow-em --confirm-full-pilot
```

これは **Python プロセスに渡された引数列**であり、`--full --allow-em --confirm-full-pilot`
の triple gate を通過した production full pilot 起動である。

---

## 3. `stdout.log` は存在し、内容は保存された runtime output と整合する

`stdout.log` は 3246 bytes、CRLF 終端の JSON 2 行からなる。

| 行 | 内容 | 対応する runner の出力箇所 |
|---|---|---|
| 1 | `{"stage": "split_preflight", "em_fits_executed": 0, "replicates": [1,2,3], "expected_test_pairs": [555,555,555], ...}` | 3 replicate 全 split の preflight 完了時点。**EM fit 実行数 0**（all-split preflight が first EM fit より前であることの記録） |
| 2 | `{"mode": "full", "em_fits_executed": 42, "selected_k_by_replicate": {"1":3,"2":3,"3":5}, "selected_k_counts": {"3":2,"5":1}, "descriptive_recovery_rate": 0.666..., "run_code_sha": "b9311e64...", ...}` | `run_full_pilot_cli()` の戻り値を `main()` が `json.dumps` したもの |

2 行目の値は `replicate_selection.csv` / `aggregate_summary.csv` / `runinfo.json` と一致する。

---

## 4. Repository evidence から**確定できたこと**（DERIVED、推測ではない）

`stdout.log` の 2 行目に含まれる `"artifacts"` フィールドは、
`run_heldout_k_selection_pilot.py` の

```python
def _require_only_expected_artifacts(out_dir: Path) -> list[str]:
    allowed = set(FULL_PILOT_ARTIFACT_NAMES) | {FULL_PILOT_STDOUT_NAME}
    present = sorted(path.name for path in out_dir.iterdir() if path.is_file())
    ...
    return present
```

が返す **実ファイルシステム列挙**である（`run_full_pilot_cli` L.3341 → `main()` L.3402 で
`json.dumps(result, sort_keys=True, allow_nan=False)` として stdout へ出力）。

保存された `stdout.log` の該当値は

```
["aggregate_summary.csv", "fit_results.csv", "manifest.csv", "replicate_selection.csv",
 "runinfo.json", "runinfo.md", "score_by_k.csv", "stdout.log"]
```

であり、**`stdout.log` 自身を含み、かつソート済み**である。

これは `runinfo.json` の `generated_artifacts`

```
["aggregate_summary.csv", "fit_results.csv", "manifest.csv", "replicate_selection.csv",
 "score_by_k.csv", "runinfo.json", "runinfo.md", "stdout.log"]
```

とは**順序が異なる別物**である。後者は
`artifacts=[*artifact_names, "runinfo.json", "runinfo.md", FULL_PILOT_STDOUT_NAME]`
という**静的な宣言リスト**であり、ファイルの存在確認を伴わない。
前者だけが実列挙である。

したがって次が言える:

> **`stdout.log` は、runner プロセスが終了する前（`_require_only_expected_artifacts` 実行時点）に、
> 出力ディレクトリ内の regular file として既に存在していた。**

この帰結として、**「プロセス終了後の事後 copy のみ」で `stdout.log` が生成された可能性は排除される。**

---

## 5. Repository evidence から**確定できないこと**

**exact capture method / outer command: `NOT RECOVERABLE FROM REPOSITORY EVIDENCE`**

- runner 自身に `stdout.log` への write 処理は **存在しない**
  （`FULL_PILOT_STDOUT_NAME` の用途は runinfo へのパス記録・report 行・許可 artifact 名のみ）。
- `runinfo.json` の `command` は **inner Python command のみ**を記録しており、outer shell 側を記録していない。
- committed script / wrapper / Makefile / CI job / PR description / report のいずれにも
  capture command の記録が無い（本 addendum 作成前の時点で committed な `stdout.log` 参照を
  すべて確認済み。いずれも path 参照・定数・テストであり、生成コマンドではない）。
- §4 の制約を満たす方式は複数あり（shell redirection `> stdout.log`、`Tee-Object`、`tee`、
  プロセス開始前にファイルを作る任意の wrapper）、**保存 artifact からは区別できない。**
- 行終端が CRLF であることは Windows 上の Python `print()` でも shell redirection でも生じるため、
  **方式を識別する証拠にならない。**

**したがって、どの方式であったかをここで推測して記録しない。**

---

## 6. 「exactly once」に関する現在の安全な表現

repository 上の今後の current claim では、次を分けて書く。

### 書いてよい

- **frozen RUN_CODE_SHA `b9311e64...` の後に、42 clean fits からなる 1 回の successful recorded execution が
  保存 artifact として確認されている。**
- 42/42 clean fit artifact が保存されている（internal retry 0 / warning 0 / Q failure 0 / NaN 0 / 非有限 0）。
- researcher の procedural record では rerun なしと記録されている。
- 独立 self-audit（artifact のみ、harness selector を import しない）は PASS、BLOCKER 0 / HIGH 0。

### 書いてはいけない

- **「削除された先行試行が存在しないことまで含めて externally proven exactly once」**
- 「repository evidence だけで exactly once が外部独立に証明されている」

理由は 2 つある。

1. §5 のとおり outer capture command が復元できない。
2. `_require_no_existing_full_artifacts()` が検査するのは `FULL_PILOT_ARTIFACT_NAMES` の 7 件のみで、
   `stdout.log` は**検査対象外**である。したがって、成果物を削除した先行試行があったとしても
   repository 上に痕跡は残らない。

**「exactly once」は researcher procedural record であり、repository-only external verification の
範囲ではない。** 両者を同一視しない。

---

## 7. この制約が無効化しないもの

§5・§6 の限定は、次のいずれも無効化しない。

| 項目 | 状態 |
|---|---|
| selected K（replicate1:3 / replicate2:3 / replicate3:5） | 不変 |
| selected-K counts `{3:2, 5:1}` / descriptive recovery rate 2/3 | 不変 |
| selector arithmetic | 独立再計算で一致（mean score 差 **0.0**、K 別集約差 1.73e-18 < tolerance 1e-12） |
| seed convention（data 41000+r / split 42000+r / model 43000+r*1000+K*10+start） | 不変・manifest と一致 |
| per-replicate hash 群（x / training_y / train_mask / test_mask / fit_provenance / target_topology / score_target / preprocessing / score_config） | 不変 |
| leakage isolation（fit 側に raw test Y が入らないこと） | Codex review PASS |
| 42 saved fit rows | 不変 |
| model lineage（`DualExpFamLSMConsistent`、experimental prototype、**本文採用不可**） | 不変 |

**stdout capture provenance は「どの外側コマンドがログを取ったか」の問題であり、
選択結果・算術・seed・hash・リーク分離のいずれにも影響しない。**

---

## 8. Finding B に対して行った修正（2026-08-31）

`tools/research_audit/audit_heldout_full_pilot.py` を **fail-closed** に強化した。
これは post-run audit hardening code であり、
**scientific execution に使われた code（RUN_CODE_SHA `b9311e64...`）とは別のコミットである。**
`run_heldout_k_selection_pilot.py` の scientific semantics は変更していない。

修正前に実在した fail-open:

| # | fail-open | 修正 |
|---|---|---|
| 1 | required artifact の存在検査が無く、欠損時は例外で落ちるか、`runinfo.md` / `stdout.log` は**そもそも検査されなかった** | 8 件の required artifact 存在を read 前に検査。欠損は BLOCKER で早期 FAIL |
| 2 | `replicate_selection.csv` が header-only でも比較 loop が 0 回で PASS | 21 行厳密・key 集合 `{1,2,3}×{1..7}`・重複禁止・replicate 内定数列の一貫性を検査 |
| 3 | `score_by_k.csv` が header-only でも PASS | 7 行厳密・K 集合 `{1..7}`・重複禁止 |
| 4 | `aggregate_summary.csv` の k_wise 行欠損・重複・pilot key 欠損/重複/余剰が素通り | 12 行厳密・k_wise 7 行・pilot key 5 件厳密・未知 section 拒否 |
| 5 | **`descriptive_recovery_rate` 欠損時に `float("nan")` となり、NaN 比較が常に False のため BLOCKER が出なかった** | 存在・有限性を先に検査してから比較 |
| 6 | runinfo の required field 検査が無い／`generated_artifacts` と実ファイルの整合を見ていない | 31 field の存在検査＋宣言と実列挙の突合 |

`tools/research_audit/test_heldout_k_selection_pilot.py` に **artifact-only の negative test を 25 件追加**した
（EM・model fit を一切呼ばない。frozen run dir を temp directory に copy して 1 箇所だけ壊す方式）。
凍結成果物に対する audit は **PASS / BLOCKER 0 / HIGH 0** のままである。

---

## 9. 意図的に修正しなかったもの（limitation）

`run_heldout_k_selection_pilot.py` の `_require_no_existing_full_artifacts()` に
`stdout.log` を**追加しない**。

理由は仮定ではなく §4 の証拠である。`stdout.log` は runner プロセス終了前に既に存在していた。
`_require_no_existing_full_artifacts()` は `run_full_pilot_cli()` の**冒頭**で実行されるため、
そこに `stdout.log` を加えていた場合、**実際に行われた 2026-08-23 の execution は開始前に BLOCK されていた。**
すなわちこの追加は過去の scientific execution semantics を変更する。

同様に `_require_only_expected_artifacts()` に required-set completeness を加える案も採らない。
同関数は run 末尾で実行されるが、`stdout.log` を必須にすると
**capture を伴わない起動（コンソール出力のまま）で 42 fits が成功しても最終段で失敗する**ことになり、
これも capture 方式を仮定した semantics 変更にあたる。

したがって **runner helper は変更せず**、completeness の fail-closed 化は
**audit layer 側でのみ**行った。runner 側の残存 limitation は次の 2 点である。

1. `_require_no_existing_full_artifacts()` は `stdout.log` の事前存在を検査しない。
2. `_require_only_expected_artifacts()` は unexpected artifact のみを拒否し、required set の完全性を検査しない。

いずれも **audit layer が fail-closed で覆っている**が、runner 単体では覆われていない。

---

## 10. 本 addendum で行っていないこと

- 42 fits の再実行 — **していない**
- EM / model fit / new seed / scientific rerun — **していない**
- result CSV・runinfo.json・runinfo.md・stdout.log の変更 — **していない**（`git diff` 空を確認）
- 2026-08-24 report の本文書き換え — **していない**
- RUN_CODE_SHA の上書き・置換 — **していない**
- frozen configuration の変更 — **していない**
- stdout capture 方式の推測による補完 — **していない**
