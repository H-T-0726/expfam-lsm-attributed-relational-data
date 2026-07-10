"""
experimental — 修論フェーズ（過分散・pair mask・共有Z ablation）用の実験的実装。

既存の安定実装（model_dual_expfam.py / model_dual_expfam_fixed.py /
utils_expfam.py）は一切変更せず、サブクラス・新規関数として追加する。

含まれるもの:
    model_dual_expfam_masked.py : pair mask（strict held-out）対応モデル
    model_dual_expfam_nb.py     : Negative Binomial (NB2, 固定 dispersion) Y モデル
    em_runner.py                : masked / NB 対応の汎用 MCEM ランナー
    eval_utils.py               : held-out 予測対数尤度・過分散診断・BIC 等
    data_generator_overdispersed.py : NB-Y 人工データ生成器
    test_experimental_models.py : スモークテスト

注意（KI-001/KI-002）:
    すべて fixed 版（DualExpFamLSMFixed, E-step 0.5 なし）の系列。
    旧版（0.5 あり）との数値混在禁止。
"""
