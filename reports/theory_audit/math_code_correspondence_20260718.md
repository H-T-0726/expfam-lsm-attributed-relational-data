# 数式・コード対応表（理論監査 2026-07-18）

対象コミット: `3fe24b6`（dirty worktree、既存ファイル無変更）。
全行番号は本監査で Read により実物と突合済み。
「一致」列: ✓=理論と一致 / △=条件付き一致 / ✗=不一致 / ?=未検証。
記法: s(η)=sigmoid、A′/A″=対数分配関数の 1・2 階微分、φ=分散パラメータ、
O=観測ペア集合（上三角）、q_i=per-node Laplace 近似事後。

| 理論上の項 | 数式 | 前提 | 実装ファイル | 関数/クラス | 行番号 | 現在の実装 | 理論との一致 | 問題点 | 必要な修正 | 検証方法 |
|---|---|---|---|---|---|---|---|---|---|---|
| Z prior | z_i ~ N(0, σ_z² I), σ_z²=1 固定 | 識別のため固定 | reproduction/src/model.py | initialize_params | 98-99, 122-130 | var_z=1.0 固定 | ✓ | scale_Z との関係（下記） | なし | — |
| Z prior 勾配項 | ∂(−ln f)/∂z_i ⊃ (1/σ_z²) z_i | — | 同上 / 全系列 | _calc_gradient Term1 | model.py 264 / fixed 57 / percolumn 124 | −(1/varz)z_i を負勾配へ | ✓ | — | — | percolumn 監査 B で数値微分一致済 |
| X 尤度 Gaussian | ln p = −(x−η)²/2σ_l² − ½ln σ_l² − ½ln2π | η=f_l^T z_i、切片なし | expfam/src/model_dual_expfam.py | calc_log_likelihood_X | 314-323 | ln2π 込み | ✓ | — | — | 監査 F（scipy 照合、percolumn 経由） |
| X 尤度 Bernoulli | x ln s + (1−x)ln(1−s) | x∈{0,1} | 同上 | calc_log_likelihood_X | 324-327 | clip η±500, S∈[1e-10,1-1e-10] | ✓ | 台違反データでは quasi | validation で台チェック（P1） | 監査 F |
| X 尤度 Poisson | xη − e^η − ln x! | x∈Z≥0 | 同上 | calc_log_likelihood_X | 328-330 | η clip[−20,10]、−ln x! は strict Q で補正 | △ | clip 域で尤度変形 | clip 発動率記録（P1） | 監査 F は非 clip 域のみ |
| X 勾配項（Term2） | F^T[(T(x_i)−A′(Fz_i))/φ] | canonical link | model_dual_expfam.py / fixed / percolumn | _calc_gradient | 139-150 / 59-68 / 126-128 | Gaussian: 1/σ_l 重み、他: 重み1、percolumn: 列別重み w_l | ✓ | — | — | 監査 B（数値微分、≤2.8e-9） |
| X 曲率項（Term2） | F^T diag[A″(Fz_i)/φ] F | 同上 | 同上 | _calc_precision_matrix | 183-194 / 99-107 / 143-145 | 同上 | ✓ | — | — | 監査 C（数値ヤコビアン、≤4.2e-10） |
| Y 尤度（family_y） | Σ_{(i,j)∈O} [T(y)η − A(η)]/φ + h 項 | η=w0+w z_i^Tz_j 対称 | model_expfam.py / masked | calc_log_likelihood_Y | 241-269 / 224-251 | 全行列×mask を ½ 和 = 上三角和 | ✓ | Gaussian-Y は ln2π 省略（strict 側で補正、旧経路は未補正） | 旧 calc_Q_dual_strict に注記 or 廃止（P1） | 手計算・eval_utils 照合 |
| Y 勾配項（Term3, fixed 系列） | w Σ_{j≠i∈O_i}[T(y_ij)−A′(η_ij)]/φ_Y z_j（1/2 なし） | §4.1 導出 | model_dual_expfam_fixed.py / masked / percolumn | _calc_gradient | 70-77 / 110-114 / 130-133 | w*(Z^T resid)、mask 乗算 | ✓ | — | — | 監査 B（Y 3 family × mask 有無） |
| Y 曲率項（Term3, fixed 系列） | w² Σ_{j≠i∈O_i} A″(η_ij)/φ_Y z_j z_j^T | 同上 | 同上 | _calc_precision_matrix | 109-113 / 132-134 / 147-149 | 同上 | ✓ | — | — | 監査 C |
| **0.5 係数（旧系列）** | 上式に 0.5 を乗じる | — | reproduction/model.py; model_expfam.py; model_dual_expfam.py | _calc_gradient / _calc_precision_matrix | 283&353; 109&135; 159&200 | 勾配・precision の両方に 0.5 | ✗（真の事後に対して）/ △（Y 尤度^{1/2} の温度緩和事後に対しては正確 [DERIVED]） | 本文採用実験（Exp1-4）はこの系列 | 修論では fixed 系列で統一（実施済み方針の継続）。旧系列は変更しない | fixed_official 再実験（実施済み）との対応表整備 |
| MATLAB 勾配（先行研究） | 同 Term3（w 必要） | — | Mato Lab Program/calcEtaNewton.m | calcGrad | 43-49（旧版 30-41） | **(Y−S)Z に w なし**、1/2 もなし | ✗（w 欠落） | 「MATLAB=正」の根拠に限定必要 | half_factor 系文書へ注記（P0-6） | 目視確認済み（本監査） |
| MATLAB 曲率（先行研究） | 同 Term3 | — | 同上 | calcAi | 56-63 | w²·s(1−s)、1/2 なし。対角除去 L.61 が二重変換 | △（1/2 なしは ✓、対角除去は ✗） | 同上 | 同上 | 同上 |
| pair mask 意味論 | (i,j)∉O は尤度・勾配・曲率・M-step から除外 | MAR+distinctness | experimental/model_dual_expfam_masked.py | set_train_mask ほか | 56-72, 93-136, 142-218, 224-251 | 対称チェック・対角 False 強制・_mask_f 乗算 | ✓ | 欠測機構は非モデル化（MCAR 分割でのみ正当） | 用語整備（P0-5） | test_masked_ignores_heldout_pairs |
| 対角除外 | y_ii を使わない | — | 全系列 | 各所 | residual[i]=0 / mask 対角 False / fill_diagonal | 一貫 | ✓ | — | — | テスト済 |
| M-step F（Gaussian X） | F = (Σ_l X^T Z_l)(Σ_l Z_l^T Z_l)^{-1} | 解析解 | reproduction/src/model.py | calc_F | 506-546 | 同式 | ✓ | — | — | 先行研究再現比較（差<0.001） |
| M-step F（非 Gaussian X） | ∇_F Q_X = (1/L)Σ_l (X−A′(Z_lF^T))^T Z_l を Adam 最大化 | 有限反復 | model_dual_expfam.py / percolumn | _calc_F_adam / _calc_F_adam_weighted | 219-268 / 162-194 | 50 反復 lr=0.01。percolumn は 1/σ_l² 重み付き | △ | Q 増加保証なし（GEM 未満の可能性） | 単調性計測（P1） | 監査 E で重み全1 時の親一致は確認済 |
| M-step Σ（Gaussian X 列） | σ_l² = (1/Ln)Σ_l Σ_i resid² | MLE | reproduction/model.py / percolumn | calc_sigma | 548-587 / 196-209 | 対角のみ、下限 1e-6 | ✓ | — | — | test_mixed_sigma_only_gaussian_cols |
| M-step w0 | ∂Q/∂w0 = (1/2Lφ)Σ_l Σ_{i≠j∈O}(T(y)−A′(η)) を Adam | 1/2 は全行列→上三角換算で**正しい** | model_expfam.py / masked | calc_w0 | 149-178 / 142-169 | /(2Lφ)、mask 乗算 | ✓ | 有限 50 反復 | 単調性計測（P1） | test（w0 一致）+ 導出 |
| M-step w | 同上 ×(z_i^Tz_j) | 同上 | 同上 | calc_w | 180-210 / 171-199 | 同上 | ✓ | 同上 | 同上 | 同上 |
| M-step σ_y（Gaussian Y） | σ_y² = mean_{l,(i,j)∈O}(y−η)² | MLE | model_expfam.py / masked | calc_sigma_y | 212-235 / 201-218 | 上三角/観測ペアのみ、下限 1e-6 | ✓ | — | — | 手計算照合 |
| Newton 更新 | z ← z − α A_i^{-1} ∇(−ln f) | mode 探索 | reproduction/src/model.py | calc_eta_newton | 360-462 | max_iter=10、α: runner から 0.5（reproduction 既定 0.01）、対称化+1e-6 正則化 | △ | 少反復・damping で mode 未収束のまま Laplace になり得る | 収束診断の記録（P1） | grad ノルム記録実験 |
| Laplace 共分散・サンプリング | z_i ~ N(mode, A_i(mode)^{-1}) | mode で評価 | 同上 | calc_eta_newton | 446-460 | 最終点で A_i 再計算後 mvn サンプル | △ | mode 未収束点まわり。ノード逐次で joint 依存無視 | 同上 | 同上 |
| MC サンプル生成 | z^{(l)} iid ~ q | MCEM | utils_expfam.py / em_runner.py | run_em_dual / run_em_experimental | 494-516 / 129-149 | 前サンプル初期値の逐次チェーン（独立でない） | ✗（iid でない） | 有効サンプル数 < L の可能性 | L・独立化の感度実験（P2） | 自己相関計測 |
| scale_Z | （モデル外）全サンプル平均二乗→1 | MATLAB 由来ヒューリスティック | reproduction/src/model.py | scale_Z | 468-504 | 毎 EM 反復適用 | ✗（尤度原理外） | MCEM 対象分布を変更。スケール尾根対策としては実務的 | 除去アブレーション（P1） | scale_Z on/off 比較 |
| Q 関数 | Q̂=(1/L)Σ_l[ln p(Z^l)+ln p(X\|Z^l)+ln p(Y_O\|Z^l)] | EM の Q の MC 近似 | utils_expfam.py / eval_utils.py | calc_Q_dual(_strict) / calc_Q_dual_strict_exp | 324-379 / 186-229 | 定数補正: Poisson 階乗・（exp 版のみ）Gaussian-Y ln2π・mixed の Poisson 列 | ✓（Q として） | **周辺尤度ではない**（H(q) 欠落、報告 §7） | ELBO 補正の実装（P1） | H(q) 計測実験 |
| BIC | −2Q̂ + p̂ ln n; p̂=kd−k(k−1)/2+Σ分散 | 報告 §7-8 | utils_expfam.py / eval_utils.py | calc_bic_dual / calc_bic_exp | 386-404 / 232-256 | w0,w,Z 数えず。n=オブジェクト数 | △（ICL 型基準としては動作。Schwarz BIC ではない） | 疎データで過大ペナルティの疑い（KI-011 の機構候補） | 基準の改称+ELBO 補正+held-out 主基準（P0-1/P1） | Cora 設定で H(q) と選択 k の関係を計測 |
| BIC 例外処理 | — | — | experimental/em_runner.py | run_em_experimental | 173-184 | except Exception: pass で NaN | ✗（運用上） | 失敗が無警告 | 警告ログ追加（P1） | — |
| Procrustes | min_R∈O(k)‖Z_est R − Z_true‖ | O(k) 不定性（報告 §6） | utils_expfam.py | procrustes_rotation | 38-43 | R=UV^T（反射込み） | ✓ | スケール・Gram 評価は別途 | Gram/リンク確率評価の追加（P2） | — |
| 点推定 Z_est | 事後平均 or mode | — | utils_expfam.py / em_runner.py | run_em_dual / predict_mu_y | 543 / 199-204 | 最後の 1 サンプル | △ | サンプリングノイズが RMSE・予測に加算 | 事後平均使用の比較（P1） | サンプル平均 vs 最終サンプル比較 |
| 生成器 Z | z_i ~ N(0,I) | — | data_generator_expfam.py | generate_dual_data | 281-283 | 生成後列 z-score | △ | 「真値」は厳密な prior サンプルでない（O(n^{-1/2})） | 文書化（P0-5） | z-score 前後比較実験（P2） |
| 生成器 F | 行ノルム √(1−uniq) | — | 同上 | 同上 | 285-290 | var_f は方向のみ寄与 | ✓（設計として） | 文書化不足 | 文書化（P0-5） | — |
| 生成器 Gaussian-X | X=ZF^T+ε 後に列 z-score | Var≈1 設計で z-score≈恒等 | 同上 | 同上 | 292-298 | 返却 F,σ は z-score 前の値 | △ | RMSE(F) に生成側ずれ混入の可能性 | 定量化実験（P2） | 正規化定数を返して補正比較 |
| 生成器 Y | 上三角生成→対称化・対角 0 | i<j モデルと一致 | 同上 | 同上 | 308-326 | ✓ | ✓ | — | — | — |
| Poisson clip | η_c=clip(η,−20,10) | 数値ガード | model_expfam.py ほか全系列 | _mean_function ほか | 57, 73, 102, 115 ほか | A′,A″,尤度すべて clip | △ | clip 域で「実装勾配 ≠ 実装尤度の勾配」（報告 §5.1-U1） | 発動率記録・整合化検討（P0-2） | clip 発動カウンタ実験 |
