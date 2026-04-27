# Section 2: 提案手法の理論的基盤 — sec2_v16
# 査読済み修正ポイント:
#   [FIX-1] 2.0 節（地図）を追加
#   [FIX-2] 生成モデルの µ_x 扱いを統一（中心化済みと明記）
#   [FIX-3] 「任意の ExpFam」→「1-パラメータ標準指数型分布族」と限定
#   [FIX-4] Toggle 1 (2022): A_i = 精度行列（2024 は共分散行列）と正確に記述
#   [FIX-5] Toggle 1 の ηi 式: (xi + µ) → (xi − µ)（PDF 抽出バグ修正）
#   [FIX-6] Toggle 1 の 2022 vs 2024 記法差異を注記

# ─── 2.0 この節の地図 ─────────────────────────────────────────────
B.append(h2("2.0 この節の地図"))
B.append(para(rt(
    "この節では、提案手法 Dual-ExpFam LSM の理論を生成モデルから出発して順に導出する。"
    "具体的には：生成モデルの定義（2.2） → complete-data log-likelihood（2.3）"
    " → MCEM の必然性（2.4） → Laplace 近似の必然性（2.5）"
    " → gradient/Hessian の一般導出（2.6） → M-step（2.7）"
    " → 3シナリオへの特殊化（2.8） → 先行研究への後退互換性確認（2.9）の順に進む。"
    "2022・2024 先行研究の詳細式はこの節末尾のトグルにまとめ、"
    "本文には比較に必要な最小限の接続のみ残す。"
)))
B.append(tbl([
    ["節","主題","ポイント"],
    ["2.1","提案の核心","X×Y を 1-パラメータ ExpFam に一般化；推論の必然性"],
    ["2.2","生成モデル","dispersion 込みの統一形で定義；µ_x 中心化の明示"],
    ["2.3","complete-data log-likelihood","prior / X / Y の3項分解"],
    ["2.4","なぜ MCEM か","E-step 積分が一般に解析困難 → MC 近似"],
    ["2.5","なぜ Laplace 近似か","posterior が非 Gaussian → MAP 二次近似"],
    ["2.6","一般導出","gradient → Hessian → A''(η)/a(φ) が曲率として登場"],
    ["2.7","M-step の導出","閉形式解 / gradient-based の区分、F の途中式"],
    ["2.8","シナリオ A/B/C への特殊化","3分布族での A''(η) と精度行列"],
    ["2.9","後退互換性","Gauss×Bern→2024 Eq.(22); Gauss×Gauss→2022 Eq.(9)"],
    ["トグル","先行研究詳細","Mikawa 2022 / 2024 の詳細導出（MCEM・式番号付き）"],
]))

# ─── 2.1 提案の核心 ─────────────────────────────────────────────
B.append(h2("2.1 提案の核心"))
B.append(callout(
    "Dual-ExpFam LSM の核心は以下の3点にある。\n\n"
    "① X と Y をともに 1-パラメータ標準指数型分布族（Bernoulli・Poisson・Gaussian など）でモデル化する。\n"
    "  （ベクトル自然母数を持つカテゴリカル分布等は本フレームワークには自動では含まれない。）\n\n"
    "② Z の事後分布 p(z_i | X, Y, θ) は、X または Y が非 Gaussian のとき\n"
    "  closed form（解析解）を持たないため、Laplace 近似（MAP 近傍の Gaussian 近似）を使う。\n\n"
    "③ Laplace 近似で得た近似事後分布からサンプリングし、Q 関数をサンプル平均で近似する。\n"
    "  これが MCEM（Monte Carlo EM）である。\n\n"
    "先行研究との関係：Mikawa 2022（continuous Y）も Mikawa 2024（binary Y）も MCEM を使う点は同じ。\n"
    "2022 は Y=Gaussian なので正確な Gaussian 事後分布を得られ Laplace 近似は不要。\n"
    "2024 は Y=Bernoulli のため非 Gaussian 事後になり Laplace 近似が必要。\n"
    "本研究はその枠組みを指数型分布族全体に体系的に一般化する。",
    "🎯", "yellow_background"
))
B.append(para(rt(
    "核心的な一般化は曲率項の置き換えにある："
    "2024 精度行列の Y 側に現れる Bernoulli 固有の s_ij(1-s_ij)（= A_Y''(η) / a_Y, Bernoulli では a_Y=1）を"
    "任意分布族の分散関数 A_Y''(η) / a_Y(φ_Y) に置き換え、かつ X 側も同様に一般化することで"
    "E-step の骨格を保ちつつ曲率・残差・M-step 更新の分布依存部分を統一的に一般化した。"
    "先行研究の詳細はこの節末尾のトグルを参照。"
)))

# ─── 2.2 生成モデル ─────────────────────────────────────────────
B.append(h2("2.2 提案手法の生成モデル"))
B.append(h3("2.2.1 指数型分布族の統一形（dispersion 込み）"))
B.append(para(rt(
    "本稿では以下の dispersion パラメータ φ を含む 1-パラメータ標準指数型分布族を扱う："
)))
B.append(eq_block(
    r"p(y;\,\eta,\varphi) = h(y,\varphi)\exp\!\left\{\frac{\eta\,T(y) - A(\eta)}{a(\varphi)}\right\}"
))
B.append(tbl([
    ["記号","意味","重要性質"],
    ["η","自然母数（natural parameter）","モデルパラメータ θ の関数"],
    ["T(y)","十分統計量","残差 T(y) − A'(η) = 実測 − 予測"],
    ["A(η)","対数分配関数","凸関数；A'(η) = E[T(Y)], A''(η) = Var[T(Y)] ≥ 0"],
    ["a(φ)","dispersion 関数","Bernoulli / Poisson は a(φ)=1; Gaussian は a(φ)=σ²"],
    ["h(y,φ)","基底測度","分布族ごとに固定"],
]))
B.append(callout(
    "a(φ) の役割：Bernoulli・Poisson は自然母数だけで分布が決まるため a(φ)=1。\n"
    "Gaussian は分散 σ² が追加パラメータとして現れ a(φ)=σ²。\n"
    "これにより Hessian の係数が統一的に A''(η)/a(φ) と書ける。\n"
    "A''(η)/a(φ) はのちに精度行列の曲率として Hessian から自然に登場する（2.6.3 節参照）。",
    "💡", "blue_background"
))
B.append(tbl([
    ["分布族","T(y)","A(η)","A'(η)","A''(η)","a(φ)"],
    ["Bernoulli","y","ln(1+e^η)","σ(η)=1/(1+e^{-η})","σ(η)(1-σ(η))","1"],
    ["Poisson","y","e^η","e^η","e^η","1"],
    ["Gaussian","y","η²/2","η","1","σ²（分散）"],
]))
B.append(h3("2.2.2 本提案の生成モデル"))
B.append(para(rt("n ノード・潜在次元 k・属性次元 d として：")))
B.append(eq_block(r"z_i \sim \mathcal{N}(0,\,\sigma_z^2 I_k), \quad i = 1,\ldots,n"))
B.append(para(rt(
    "属性データ（第 j 特性量）— F の行を f_j ∈ R^k として。"
    "Gaussian X では後述のように中心化済み入力を使う："
)))
B.append(eq_block(
    r"p(x_{ij}\mid z_i;\,\varphi_X) = h_X(x_{ij},\varphi_X)\exp\!\left\{\frac{\eta_{ij}^X T_X(x_{ij}) - A_X(\eta_{ij}^X)}{a_X(\varphi_X)}\right\},"
    r"\quad \eta_{ij}^X = f_j^\top z_i"
))
B.append(para(rt("関係データ（i < j の対）：")))
B.append(eq_block(
    r"p(y_{ij}\mid z_i,z_j;\,\varphi_Y) = h_Y(y_{ij},\varphi_Y)\exp\!\left\{\frac{\eta_{ij}^Y T_Y(y_{ij}) - A_Y(\eta_{ij}^Y)}{a_Y(\varphi_Y)}\right\},"
    r"\quad \eta_{ij}^Y = w_0 + w\,z_i^\top z_j"
))
B.append(callout(
    "µ_x（平均オフセット）の扱い：\n"
    "2022・2024 の生成モデルは xi = F zi + µ_x + εi（Gaussian X）として µ_x を陽に持つ。\n"
    "本提案では Gaussian X のとき、入力を中心化済み x̃_i = xi − µ_x と見なして"
    " η^X_{ij} = f_j^T z_i を適用する（µ_x は M-step で推定）。\n"
    "非 Gaussian X（Bernoulli・Poisson）では T(x)=x かつ A'(η) が自然母数の関数であり、"
    " µ_x は通常定義しない。\n"
    "この中心化前提は生成モデルと M-step（2.7 節）の式を一貫させるための仮定である。",
    "⚠️", "yellow_background"
))
B.append(tbl([
    ["パラメータ","記号","備考"],
    ["潜在変数","z_i ∈ R^k","ノード i の潜在座標（推論対象）"],
    ["因子行列","F ∈ R^{d×k}","X の負荷行列（M-step で更新）"],
    ["関係パラメータ","w_0 ∈ R, w ∈ R","バイアスと内積スケール"],
    ["潜在空間スケール","σ_z² > 0","事前分布の分散"],
    ["X 側 dispersion","φ_X","Gaussian: φ_X=σ²_X, 非 Gaussian: a(φ_X)=1"],
    ["Y 側 dispersion","φ_Y","Gaussian: φ_Y=σ²_Y, 非 Gaussian: a(φ_Y)=1"],
]))
B.append(callout(
    "先行研究との対応：\n"
    "Mikawa 2022（continuous relation）は Y=Gaussian × X=Gaussian, w=1, w_0=0 の特殊例。\n"
    "Mikawa 2024（binary relation）は Y=Bernoulli × X=Gaussian の特殊例。\n"
    "本提案はこれを X・Y ともに 1-パラメータ標準指数型分布族に一般化したものである。",
    "💡", "blue_background"
))

# ─── 2.3 complete-data log-likelihood ─────────────────────────
B.append(h2("2.3 complete-data log-likelihood の分解"))
B.append(para(rt(
    "Z を観測済みと仮定した完全データ対数尤度を prior / X / Y の3項に分解する。"
    "ここでは Laplace 近似も MCEM も使わず、提案手法そのものの式として立てる。"
)))
B.append(eq_block(
    r"\ln p(X,Y,Z\mid\theta)"
    r"= \underbrace{\sum_i\ln p(z_i)}_{\text{prior}}"
    r"+ \underbrace{\sum_{i=1}^n\sum_{j=1}^d \ln p(x_{ij}\mid z_i)}_{\text{X term}}"
    r"+ \underbrace{\sum_{i<j} \ln p(y_{ij}\mid z_i,z_j)}_{\text{Y term}}"
))
B.append(para(rt("各項を展開すると：")))
B.append(eq_block(
    r"\text{prior} = -\frac{nk}{2}\ln(2\pi\sigma_z^2) - \frac{1}{2\sigma_z^2}\sum_i\|z_i\|^2"
))
B.append(eq_block(
    r"\text{X term} = \sum_{i,j}\frac{\eta_{ij}^X T_X(x_{ij}) - A_X(\eta_{ij}^X)}{a_X(\varphi_X)} + \text{const}_X"
))
B.append(eq_block(
    r"\text{Y term} = \sum_{i<j}\frac{\eta_{ij}^Y T_Y(y_{ij}) - A_Y(\eta_{ij}^Y)}{a_Y(\varphi_Y)} + \text{const}_Y"
))
B.append(callout(
    "なぜ i < j のみ和をとるか：y_ij は無向グラフの対称関係（y_ij = y_ji）なので\n"
    "i < j の「上半行列」だけ和をとれば全ペアをちょうど一度数えられる。\n"
    "これは 2022 Eq.(5) / 2024 Eq.(16) の {∏_{i≠j} p(y_ij)}^{1/2} 表現と等価である。\n"
    "すなわち Y 側対数尤度の 1/2 因子は i < j の和と同じ意味を持ち、\n"
    "precision matrix の Y 側に w²/(2 a_Y) という係数として現れる。",
    "💡", "blue_background"
))
B.append(para(rt(
    "Z は潜在変数なので観測されない。EM では Q 関数（Z の事後分布による期待値）を最大化する。"
    "次節でこの期待値がなぜ厳密計算できないかを示す。"
)))

# ─── 2.4 なぜ MCEM が必要か ─────────────────────────────────────
B.append(h2("2.4 なぜ MCEM（Monte Carlo EM）が必要か"))
B.append(h3("2.4.1 EM の Q 関数"))
B.append(para(rt(
    "EM アルゴリズムは完全データ対数尤度の事後期待値（Q 関数）を最大化する："
)))
B.append(eq_block(
    r"Q(\theta;\theta^{\mathrm{old}}) "
    r"= \mathbb{E}_{p(Z\mid X,Y,\theta^{\mathrm{old}})}\!\bigl[\ln p(X,Y,Z\mid\theta)\bigr]"
    r"= \int p(Z\mid X,Y,\theta^{\mathrm{old}})\ln p(X,Y,Z\mid\theta)\,dZ"
))
B.append(h3("2.4.2 なぜ厳密計算できないか"))
B.append(para(rt(
    "この積分を解析的に評価できるのは、事後分布 p(Z|X,Y,θ) が既知の閉じた形になる特殊な場合だけである。"
    "X も Y も Gaussian のとき（2022 の設定）は事後分布が Gaussian になるので積分可能。"
    "しかし Y が Bernoulli や Poisson のとき、A_Y(η) が非二次形式であり事後分布が非 Gaussian になる"
    "（詳細は 2.5 節）。"
    "2022・2024・本提案はいずれも Q 関数を解析的に閉じた形で計算することはできないため、"
    "すべて MCEM を採用している（2022: Eq.(3)(4)(6)、2024: Eq.(17)(18)）。"
)))
B.append(callout(
    "直感：本来は posterior p(Z|X,Y) に関する期待値を厳密に積分したいが、\n"
    "積分が解析的に計算できないため、posterior から L 個のサンプル Z^(l) を引いて\n"
    "その平均で期待値を近似する。これが Monte Carlo EM（MCEM）である。\n"
    "L が大きいほど近似精度が上がる。本実装では L=5 を採用。",
    "💡", "blue_background"
))
B.append(h3("2.4.3 MCEM による Q の近似"))
B.append(para(rt("近似事後分布 q(Z) から L 個のサンプル Z^{(1)}, ..., Z^{(L)} を引き：")))
B.append(eq_block(
    r"\hat{Q}(\theta;\theta^{\mathrm{old}}) "
    r"= \frac{1}{L}\sum_{l=1}^L \ln p\!\bigl(X,Y,Z^{(l)}\mid\theta\bigr)"
    r"= \frac{1}{L}\sum_{l=1}^L\bigl[\ln p(Z^{(l)}|\theta)"
    r"+ \ln p(X|Z^{(l)};\theta) + \ln p(Y|Z^{(l)};\theta)\bigr]"
))
B.append(para(rt(
    "L → ∞ のとき MCEM の Q は真の Q に確率収束する（一般的 MCEM の性質）。"
    "2.5 節と 2.6 節で近似事後分布 q(Z) の構築方法（Laplace 近似）を説明する。"
)))

# ─── 2.5 なぜ Laplace 近似が必要か ─────────────────────────────
B.append(h2("2.5 なぜ Laplace 近似が必要か"))
B.append(h3("2.5.1 事後分布の Bayes 表現"))
B.append(para(rt("z_i に条件付けた事後分布は Bayes の定理から：")))
B.append(eq_block(
    r"p(z_i \mid X,Y,\theta) \propto p(z_i)\,\prod_{j=1}^d p(x_{ij}\mid z_i)"
    r"\cdot\prod_{j\neq i}p(y_{ij}\mid z_i,z_j)"
))
B.append(para(rt("z_i に関する条件付き対数事後分布（定数を除く）：")))
B.append(eq_block(
    r"\ln p(z_i\mid \cdots) = "
    r"-\frac{\|z_i\|^2}{2\sigma_z^2}"
    r"+ \sum_{j=1}^d\frac{f_j^\top z_i\cdot T_X(x_{ij}) - A_X(f_j^\top z_i)}{a_X(\varphi_X)}"
    r"+ \sum_{j\neq i}\frac{\eta_{ij}^Y T_Y(y_{ij}) - A_Y(\eta_{ij}^Y)}{a_Y(\varphi_Y)}"
    r"+ \mathrm{const}"
))
B.append(h3("2.5.2 なぜ Gaussian posterior が閉じないか"))
B.append(para(rt(
    "事前分布項は z_i の二次形式であり Gaussian 構造を持つ。"
    "しかし A_X(f_j^T z_i) と A_Y(η^Y_{ij}) は一般に非線形関数であり、"
    "z_i の二次形式にならない："
)))
B.append(tbl([
    ["分布族","A(η)","A(f_j^T z_i) の z_i 依存性","posterior の形状"],
    ["Gaussian","η²/2","(f_j^T z_i)²/2 — 二次形式","Gaussian posterior ✓"],
    ["Bernoulli","ln(1+e^η)","ln(1+exp(f_j^T z_i)) — 非二次","Gaussian posterior ✗"],
    ["Poisson","e^η","exp(f_j^T z_i) — 非二次","Gaussian posterior ✗"],
]))
B.append(para(rt(
    "Y=Gaussian の場合（2022）は全項が二次形式になるため"
    "Gaussian 事後分布を閉じた形で求められる。"
    "Y=Bernoulli（2024）や Y=Poisson では非 Gaussian 項が入るため閉じない。"
)))
B.append(callout(
    "直感：posterior 全体は複雑な非 Gaussian 分布だが、\n"
    "最も確からしい点（MAP 推定値 z_i*）の近くでは山の形を二次式で近似できる。\n"
    "その二次式に対応する Gaussian が Laplace 近似である。",
    "💡", "blue_background"
))
B.append(h3("2.5.3 Laplace 近似の手順"))
B.append(para(rt("STEP 1：Newton 法で MAP 推定値 z_i* を求める：")))
B.append(eq_block(r"z_i^* = \arg\max_{z_i}\ln p(z_i\mid X,Y,\theta)"))
B.append(para(rt("STEP 2：z_i* まわりで対数事後分布を二次展開する。精度行列 Λ_i = −∇²_{z_i} ln p を用いて：")))
B.append(eq_block(
    r"\ln p(z_i\mid\cdots) \approx \mathrm{const} - \tfrac{1}{2}(z_i - z_i^*)^\top \Lambda_i (z_i - z_i^*)"
))
B.append(para(rt("STEP 3：これは z_i ∼ N(z_i*, Λ_i^{-1}) と一致するので、近似事後分布を得る：")))
B.append(eq_block(r"q(z_i) = \mathcal{N}\!\bigl(z_i^*,\;\Lambda_i^{-1}\bigr)"))
B.append(para(rt(
    "精度行列 Λ_i の具体形（A''(η)/a(φ) を含む）を次節 2.6 で Hessian から導出する。"
    "2024 論文の Ai は Λ_i^{-1}（共分散行列）に相当する（Eq.(22) 参照）。"
)))

# ─── 2.6 提案手法の一般導出 ─────────────────────────────────────
B.append(h2("2.6 提案手法の一般導出"))
B.append(callout(
    "この節が本文の最重要部分。\n"
    "gradient → Hessian の順に導出し、Hessian から A''(η)/a(φ) が自然に現れることを示す。",
    "📌", "gray_background"
))

B.append(h3("2.6.1 条件付き対数事後分布"))
B.append(eq_block(
    r"\ln p(z_i\mid\cdots) = "
    r"-\frac{\|z_i\|^2}{2\sigma_z^2}"
    r"+ \sum_{j=1}^d \frac{f_j^\top z_i\cdot T_X(x_{ij}) - A_X(f_j^\top z_i)}{a_X(\varphi_X)}"
    r"+ \sum_{j\neq i}\frac{(w_0+w z_i^\top z_j) T_Y(y_{ij}) - A_Y(w_0+w z_i^\top z_j)}{a_Y(\varphi_Y)}"
    r"+ \mathrm{const}"
))

B.append(h3("2.6.2 gradient の導出（MAP を求めるための Newton 更新に使う）"))
B.append(para(rt("事前分布項：")))
B.append(eq_block(r"\nabla_{z_i}^{\mathrm{prior}} = -\frac{z_i}{\sigma_z^2}"))
B.append(para(rt(
    "X 項：chain rule から ∂(f_j^T z_i)/∂z_i = f_j を使う。"
    "Gaussian X のとき T_X(x_{ij})=x_{ij}（中心化済み）・A_X'(f_j^T z_i)=f_j^T z_i（予測値）："
)))
B.append(eq_block(
    r"\nabla_{z_i}^X"
    r"= \frac{1}{a_X(\varphi_X)}\sum_{j=1}^d \bigl[T_X(x_{ij}) - A_X'(f_j^\top z_i)\bigr]f_j"
    r"= \frac{1}{a_X(\varphi_X)}F^\top\!\bigl[T_X(x_i) - A_X'(Fz_i)\bigr]"
))
B.append(callout(
    "ベクトル記法の確認：T_X(x_i) = (T_X(x_{i1}), ..., T_X(x_{id}))^T（d 次元ベクトル）、\n"
    "A_X'(Fz_i) = (A_X'(f_1^T z_i), ..., A_X'(f_d^T z_i))^T。\n"
    "Gaussian X: T_X(x_{ij})=x_{ij}, A_X'(f_j^T z_i)=f_j^T z_i → 残差 = x_{ij} - f_j^T z_i。\n"
    "Poisson X: T_X(x_{ij})=x_{ij}, A_X'(f_j^T z_i)=exp(f_j^T z_i) → 残差 = x_{ij} - λ_{ij}。",
    "💡", "blue_background"
))
B.append(para(rt(
    "Y 項：chain rule から ∂(w_0 + w z_i^T z_j)/∂z_i = w z_j を使う。"
    "Y 側の i<j 対称和をとると ∂/∂z_i [Σ_{i<j} ...] = Σ_{j<i} (Y 側) + Σ_{j>i} (Y 側)。"
    "対称性 y_ij=y_ji、η^Y_ij=η^Y_ji から全 j≠i の和を 1/2 倍する等価な表記を使う："
)))
B.append(eq_block(
    r"\nabla_{z_i}^Y"
    r"= \frac{w}{2\,a_Y(\varphi_Y)}\sum_{j\neq i}\bigl[T_Y(y_{ij}) - A_Y'(\eta_{ij}^Y)\bigr]z_j"
))
B.append(para(rt("合成 gradient：")))
B.append(eq_block(
    r"g_i = -\frac{z_i}{\sigma_z^2}"
    r"+ \frac{F^\top[T_X(x_i)-A_X'(Fz_i)]}{a_X(\varphi_X)}"
    r"+ \frac{w}{2\,a_Y(\varphi_Y)}\sum_{j\neq i}\bigl[T_Y(y_{ij})-A_Y'(\eta_{ij}^Y)\bigr]z_j"
))

B.append(h3("2.6.3 Hessian と精度行列 — A''(η)/a(φ) の自然な登場"))
B.append(para(rt("事前分布項の Hessian：")))
B.append(eq_block(r"-\nabla^2_{z_i}[\text{prior}] = \frac{1}{\sigma_z^2}I"))
B.append(para(rt(
    "X 項の Hessian：gradient を z_i でもう一度微分する。"
    "chain rule から ∂/∂z_i [A_X'(f_j^T z_i)] = A_X''(f_j^T z_i) f_j より："
)))
B.append(eq_block(
    r"-\nabla^2_{z_i}[\text{X term}]"
    r"= \frac{1}{a_X(\varphi_X)}\sum_{j=1}^d A_X''(f_j^\top z_i)\,f_jf_j^\top"
    r"= \frac{1}{a_X(\varphi_X)}F^\top\mathrm{diag}\!\bigl[A_X''(Fz_i)\bigr]F"
))
B.append(callout(
    "A_X''(η)/a_X(φ_X) が Hessian から自然に出てくる。\n"
    "A_X''(η) = Var[T_X(X)|η] は「η における予測の不確かさ（曲率）」を表す。\n"
    "曲率が大きいほど精度行列への寄与が大きく、z_i の推定をより強く引き寄せる。\n\n"
    "Bernoulli: A_X''=σ(η)(1-σ(η)) ≤ 0.25、a(φ)=1  →  寄与は σ(1-σ)\n"
    "Poisson:   A_X''=e^η、a(φ)=1               →  寄与は λ（レート）\n"
    "Gaussian:  A_X''=1、a(φ)=σ²               →  寄与は 1/σ²（Fisher 情報量に一致）",
    "💡", "blue_background"
))
B.append(para(rt(
    "Y 項の Hessian：chain rule: ∂A_Y'(η)/∂z_i = A_Y''(η) · w z_j より："
)))
B.append(eq_block(
    r"-\nabla^2_{z_i}[\text{Y term}]"
    r"= \frac{w^2}{2\,a_Y(\varphi_Y)}\sum_{j\neq i}A_Y''(\eta_{ij}^Y)\,z_jz_j^\top"
))

B.append(h3("2.6.4 精度行列 Λ_i の完成"))
B.append(eq_block(
    r"\Lambda_i = \underbrace{\frac{1}{\sigma_z^2}I}_{\text{prior}}"
    r"+ \underbrace{\frac{1}{a_X(\varphi_X)}F^\top\mathrm{diag}\!\bigl[A_X''(Fz_i)\bigr]F}_{\text{X term}}"
    r"+ \underbrace{\frac{w^2}{2\,a_Y(\varphi_Y)}\sum_{j\neq i}A_Y''(\eta_{ij}^Y)\,z_jz_j^\top}_{\text{Y term}}"
))
B.append(callout(
    "Λ_i は半正定値行列（A''(η) ≥ 0 の性質と σ_z² > 0 による正則化）。\n"
    "3項はそれぞれ：事前分布の等方的正則化 + X の情報 + Y の情報。\n"
    "A''(η)/a(φ) が大きい → その点での曲率が大きい → z_i の推定に大きく寄与する。",
    "💡", "blue_background"
))
B.append(h3("2.6.5 Laplace 近似事後分布"))
B.append(eq_block(
    r"q(z_i) = \mathcal{N}\!\bigl(z_i^*,\;\Lambda_i^{-1}\bigr),\quad"
    r"z_i^* = \arg\max_{z_i}\ln p(z_i\mid X,Y,\theta)"
))
B.append(para(rt(
    "この Laplace 事後分布からサンプリングして MCEM に使う（2.4 節参照）。"
    "2024 の Ai は Λ_i^{-1}（共分散行列）に相当（Eq.(22)）。"
    "2022 の Ai は Λ_i そのもの（精度行列）として定義される（Eq.(9)）— 記法の違いに注意。"
)))

# ─── 2.7 M-step の導出 ─────────────────────────────────────────
B.append(h2("2.7 M-step の導出"))
B.append(para(rt(
    "MCEM サンプル Z^{(1)}, ..., Z^{(L)} を使って Q̂ を θ について最大化する。"
    "パラメータごとに閉形式解が得られるものと gradient-based update が必要なものに分かれる。"
)))
B.append(h3("2.7.1 Gaussian X のとき — F と σ²_X の閉形式解"))
B.append(para(rt(
    "Gaussian X（中心化済み入力 x̃_i = xi - µ_x）のとき X 側 Q̂ は"
    " −(1/(2σ²_X)) Σ_{i,j} (x̃_{ij} - f_j^T z_i)² + const。"
    "Q̂ を F で偏微分してゼロとおく："
)))
B.append(eq_block(
    r"\frac{\partial\hat{Q}}{\partial F} = \frac{1}{\sigma_X^2}"
    r"\left[\frac{1}{L}\sum_{l=1}^L\sum_{i=1}^n\tilde{x}_i z_i^{(l)\top}"
    r"- F\cdot\frac{1}{L}\sum_{l=1}^L\sum_{i=1}^n z_i^{(l)}z_i^{(l)\top}\right] = 0"
))
B.append(para(rt("F について解くと閉形式解（2024 Eq.(10) / 2022 Eq.(12) に相当）：")))
B.append(eq_block(
    r"F = \left(\frac{1}{L}\sum_{l=1}^L\sum_{i=1}^n\tilde{x}_i z_i^{(l)\top}\right)"
    r"\left(\frac{1}{L}\sum_{l=1}^L\sum_{i=1}^n z_i^{(l)}z_i^{(l)\top}\right)^{-1}"
    r",\quad \tilde{x}_i = x_i - \mu_x"
))
B.append(para(rt("µ_x の更新（2024 Eq.(11) / 2022 Eq.(13) に相当）：")))
B.append(eq_block(
    r"\mu_x \leftarrow \frac{1}{Ln}\sum_{l=1}^L\sum_{i=1}^n\bigl(x_i - Fz_i^{(l)}\bigr)"
))
B.append(para(rt("σ²_X の更新（2024 Eq.(12) / 2022 Eq.(14) に相当）：")))
B.append(eq_block(
    r"\sigma_X^2 \leftarrow \frac{1}{Lnd}\sum_{l=1}^L\sum_{i=1}^n\|x_i - Fz_i^{(l)} - \mu_x\|^2"
))
B.append(h3("2.7.2 Gaussian Y のとき — σ²_Y の閉形式解"))
B.append(para(rt(
    "Gaussian Y のとき Y 側 Q̂ は −(1/(2σ²_Y)) Σ_{i<j} (y_{ij} - η^Y_{ij})² + const。"
    "σ²_Y で偏微分してゼロとおくと："
)))
B.append(eq_block(
    r"\sigma_Y^2 \leftarrow \frac{1}{Lh}\sum_{l=1}^L\sum_{i<j}"
    r"\bigl(y_{ij} - w_0 - w z_i^{(l)\top}z_j^{(l)}\bigr)^2"
    r",\quad h = \binom{n}{2}"
))
B.append(h3("2.7.3 非 Gaussian 分布族のとき — gradient-based update"))
B.append(para(rt(
    "Bernoulli・Poisson の X や Y については A_X'(η) / A_Y'(η) が非線形なため、"
    "F・w_0・w の閉形式解は一般に得られない。"
    "Q̂ の勾配を計算して Adam 等の勾配法で更新する。"
)))
B.append(para(rt("w_0, w の勾配（2024 Eq.(24)(25) の一般化）：")))
B.append(eq_block(
    r"\frac{\partial\hat{Q}}{\partial w_0}"
    r"= \frac{1}{L}\sum_{l=1}^L\sum_{i<j}\frac{T_Y(y_{ij}) - A_Y'(\eta_{ij}^{Y(l)})}{a_Y(\varphi_Y)}"
))
B.append(eq_block(
    r"\frac{\partial\hat{Q}}{\partial w}"
    r"= \frac{1}{L}\sum_{l=1}^L\sum_{i<j}\frac{T_Y(y_{ij}) - A_Y'(\eta_{ij}^{Y(l)})}{a_Y(\varphi_Y)}"
    r"\cdot z_i^{(l)\top}z_j^{(l)}"
))
B.append(callout(
    "Bernoulli Y のとき T_Y(y)=y, A_Y'(η)=σ(η)=s, a_Y=1 → 残差 = y_ij - s_ij。\n"
    "上式は i<j の和（本稿の規約）で記述。2024 Eq.(24)(25) は j≠i の和で記述しており\n"
    "両者は等価（y_ij=y_ji, s_ij=s_ji の対称性による）。",
    "✅", "green_background"
))
B.append(tbl([
    ["パラメータ","更新方法","条件"],
    ["F","閉形式 MLE（最小二乗）","Gaussian X のとき（2024 Eq.(10) / 2022 Eq.(12)）"],
    ["F","Adam 勾配法","Bernoulli / Poisson X のとき"],
    ["µ_x（平均）","閉形式 MLE","Gaussian X のとき"],
    ["σ²_X（分散）","閉形式 MLE","Gaussian X のとき"],
    ["w_0, w","Adam 勾配法","全分布族共通（2024 Eq.(24)(25) に相当）"],
    ["σ²_Y（分散）","閉形式 MLE","Gaussian Y のとき"],
    ["σ²_z（潜在）","閉形式 MLE","全分布族共通"],
]))

# ─── 2.8 シナリオ A/B/C への特殊化 ─────────────────────────────
B.append(h2("2.8 シナリオ A/B/C への特殊化"))
B.append(para(rt(
    "一般形に各分布族の A'(η), A''(η), a(φ) を代入して3シナリオの精度行列と gradient を求める。"
)))
B.append(h3("2.8.1 Scenario A：Poisson-X × Bernoulli-Y"))
B.append(callout("カウント属性 × 2値ネットワーク（例：購買数 × 友人関係有無）。", "📋", "gray_background"))
B.append(eq_block(
    r"x_{ij}\sim\mathrm{Pois}(e^{f_j^\top z_i}),\quad"
    r"y_{ij}\sim\mathrm{Bern}\!\bigl(\sigma(w_0+w z_i^\top z_j)\bigr)"
))
B.append(para(rt("A_X''（Poisson, a_X=1）・A_Y''（Bernoulli, a_Y=1）：")))
B.append(eq_block(
    r"A_X''(f_j^\top z_i)=e^{f_j^\top z_i},\quad a_X=1"
    r";\quad A_Y''(\eta_{ij}^Y)=s_{ij}(1-s_{ij}),\quad a_Y=1"
))
B.append(para(rt("精度行列：")))
B.append(eq_block(
    r"\Lambda_i^{[A]} = \frac{1}{\sigma_z^2}I"
    r"+ F^\top\mathrm{diag}(e^{Fz_i})F"
    r"+ \frac{w^2}{2}\sum_{j\neq i}s_{ij}(1-s_{ij})\,z_jz_j^\top"
))
B.append(para(rt("gradient：")))
B.append(eq_block(
    r"g_i^{[A]} = -\frac{z_i}{\sigma_z^2}"
    r"+ F^\top(x_i - e^{Fz_i})"
    r"+ \frac{w}{2}\sum_{j\neq i}(y_{ij}-s_{ij})\,z_j"
))

B.append(h3("2.8.2 Scenario B：Gaussian-X × Poisson-Y"))
B.append(callout("連続属性 × カウントネットワーク（例：特徴量 × 共著回数）。", "📋", "gray_background"))
B.append(eq_block(
    r"x_{ij}\sim\mathcal{N}(f_j^\top z_i,\,\sigma_X^2),\quad"
    r"y_{ij}\sim\mathrm{Pois}(e^{w_0+w z_i^\top z_j})"
))
B.append(para(rt("A_X''（Gaussian, a_X=σ²_X）・A_Y''（Poisson, a_Y=1）：")))
B.append(eq_block(
    r"A_X''=1,\quad a_X=\sigma_X^2"
    r"\;\Rightarrow\;\frac{F^\top\mathrm{diag}[A_X'']F}{a_X}=\frac{F^\top F}{\sigma_X^2}"
    r";\quad A_Y''(\eta_{ij}^Y)=\lambda_{ij}=e^{\eta_{ij}^Y},\quad a_Y=1"
))
B.append(para(rt("精度行列：")))
B.append(eq_block(
    r"\Lambda_i^{[B]} = \frac{1}{\sigma_z^2}I"
    r"+ \frac{1}{\sigma_X^2}F^\top F"
    r"+ \frac{w^2}{2}\sum_{j\neq i}\lambda_{ij}\,z_jz_j^\top"
))
B.append(para(rt("gradient：")))
B.append(eq_block(
    r"g_i^{[B]} = -\frac{z_i}{\sigma_z^2}"
    r"+ \frac{F^\top(x_i - Fz_i - \mu_x)}{\sigma_X^2}"
    r"+ \frac{w}{2}\sum_{j\neq i}(y_{ij}-\lambda_{ij})\,z_j"
))

B.append(h3("2.8.3 Scenario C：Bernoulli-X × Gaussian-Y"))
B.append(callout("2値属性 × 連続ネットワーク（例：バイナリ特徴 × 類似度行列）。", "📋", "gray_background"))
B.append(eq_block(
    r"x_{ij}\sim\mathrm{Bern}\!\bigl(\sigma(f_j^\top z_i)\bigr),\quad"
    r"y_{ij}\sim\mathcal{N}(w_0+w z_i^\top z_j,\,\sigma_Y^2)"
))
B.append(para(rt("A_X''（Bernoulli, a_X=1）・A_Y''（Gaussian, a_Y=σ²_Y）：")))
B.append(eq_block(
    r"A_X''(f_j^\top z_i)=p_{ij}(1-p_{ij})\le 0.25,\quad a_X=1"
    r";\quad A_Y''=1,\quad a_Y=\sigma_Y^2"
    r"\;\Rightarrow\;\frac{w^2}{2a_Y}\sum_{j\neq i}z_jz_j^\top=\frac{w^2}{2\sigma_Y^2}\sum_{j\neq i}z_jz_j^\top"
))
B.append(para(rt("精度行列：")))
B.append(eq_block(
    r"\Lambda_i^{[C]} = \frac{1}{\sigma_z^2}I"
    r"+ F^\top\mathrm{diag}[p_{ij}(1-p_{ij})]F"
    r"+ \frac{w^2}{2\sigma_Y^2}\sum_{j\neq i}z_jz_j^\top"
))
B.append(callout(
    "Scenario C の特徴：Y=Gaussian の A_Y''=1（定数）は n-1 個の j に加算されて O(n) にスケール。\n"
    "X=Bernoulli の A_X'' ≤ 0.25 は d 次元で加算されても O(0.25 d) に留まる。\n"
    "n=150 の設定では Y 項が X 項を圧倒し、d を増やしても Z の推定精度に影響しにくい。\n"
    "これが Exp3-d のフラット現象（Section 6.2）の数理的説明である。",
    "⚠️", "yellow_background"
))
B.append(para(rt("gradient：")))
B.append(eq_block(
    r"g_i^{[C]} = -\frac{z_i}{\sigma_z^2}"
    r"+ F^\top(x_i - p_i)"
    r"+ \frac{w}{2\sigma_Y^2}\sum_{j\neq i}(y_{ij}-\hat{y}_{ij})\,z_j"
))

# ─── 2.9 後退互換性 ─────────────────────────────────────────────
B.append(h2("2.9 後退互換性（先行研究との整合性確認）"))
B.append(para(rt(
    "Dual-ExpFam の一般形に特定の分布族を代入すると先行研究に正確に帰着することを確認する。"
)))
B.append(tbl([
    ["代入","Λ_i の X 側","Λ_i の Y 側","帰着先"],
    ["Gauss-X（a_X=σ²_X, A_X''=1）× Bern-Y（a_Y=1, A_Y''=s(1-s)）",
     "(1/σ²_X) F^T F = F^T Σ^{-1} F",
     "(w²/2) Σ s_ij(1-s_ij) z_j z_j^T",
     "2024 Eq.(22) の A_i^{-1} と完全一致"],
    ["Gauss-X（a_X=σ²_X, A_X''=1）× Gauss-Y（a_Y=σ²_Y, A_Y''=1）, w=1, w_0=0",
     "F^T Σ^{-1} F",
     "(1/(2σ²_Y)) Σ z_j z_j^T",
     "2022 Eq.(9) の精度行列 A_i に帰着"],
]))
B.append(callout(
    "Gauss-X × Bern-Y（→ 2024 Eq.(22)）の確認：\n"
    "A_X''=1, a_X=σ²_X → (1/a_X) F^T diag[1] F = F^T Σ^{-1} F  ✓\n"
    "A_Y''=s_ij(1-s_ij), a_Y=1 → (w²/2) Σ s_ij(1-s_ij) z_j z_j^T  ✓\n"
    "2024 Eq.(22): Ai = (1/σ²_z I + F^T Σ^{-1} F + (w²/2) Σ s_ij(1-s_ij) z_j z_j^T)^{-1}\n"
    "  ここで Ai は共分散行列（= Λ_i^{-1}）。本提案の Λ_i^{-1} と完全一致。\n\n"
    "Gauss-X × Gauss-Y, w=1, w_0=0（→ 2022 Eq.(9)）の確認：\n"
    "A_Y''=1, a_Y=σ²_Y → (w²/2σ²_Y) Σ z_j z_j^T → w=1: (1/(2σ²_Y)) Σ z_j z_j^T  ✓\n"
    "2022 Eq.(9): Ai = {(1/σ²_z)I + (1/(2σ²_y)) Σ z_j z_j^T + F^T Σ^{-1} F}\n"
    "  ここで Ai は精度行列（= Λ_i）。本提案の Λ_i と完全一致。\n\n"
    "注意：2022 と 2024 では Ai の定義が逆（精度 vs 共分散）。本提案は Λ_i=精度行列に統一。",
    "✅", "green_background"
))

# ─── トグル：先行研究詳細 ─────────────────────────────────────────
B.append(toggle("▶ 先行研究詳細 (1)：Mikawa 2022 [paper/2.pdf] — 連続値関係モデルの導出", [
    callout(
        "以下は paper/2.pdf を直接参照した記述である。\n"
        "重要：2022 論文も MCEM を採用している（Eq.(3)(4)(6)）。\n"
        "Y = Gaussian であるため事後分布が正確な Gaussian になり Laplace 近似は不要。\n"
        "ただし MCEM 自体は使用している点に注意。\n\n"
        "記法注意：2022 の Ai は精度行列（= Λ_i）として定義される。\n"
        "共分散行列は Ai^{-1}。2024 の Ai（共分散行列）とは定義が逆。",
        "📌", "gray_background"
    ),
    h3("生成モデル（2022 Eq.(1)(2)）"),
    eq_block(
        r"x_i = Fz_i + \mu + \varepsilon_i,\quad \varepsilon_i\sim\mathcal{N}(0,\Sigma)"
        r";\quad y_{ij} = z_i^\top z_j + e_{ij},\quad e_{ij}\sim\mathcal{N}(0,\sigma_y^2)"
    ),
    para(rt("Y の線形モデル y_ij = z_i^T z_j + e_ij は w_0=0, w=1 の Gaussian ノイズモデル。")),
    h3("尤度関数（2022 Eq.(5)）"),
    eq_block(
        r"\mathcal{L} = \prod_{i=1}^n p(x_i|F,z_i,\mu,\Sigma)"
        r"\left\{\prod_{i\neq j}p(y_{ij}|z_i,z_j,\sigma_y^2)\right\}^{1/2}"
    ),
    para(rt("{...}^{1/2} は対称ペア i<j を一度だけ数える等価な表現。"
            "対数をとると (1/2) Σ_{i≠j} ln p(y_ij|...) = Σ_{i<j} ln p(y_ij|...) と等価。")),
    h3("MCEM（2022 Eq.(3)(4)(6)）"),
    eq_block(r"Q(\theta,\theta^{\mathrm{old}}) = \int p(Z|X,\theta^{\mathrm{old}})\ln p(Z,X|\theta)\,dZ \quad \text{[Eq.(3)]}"),
    eq_block(r"Q \approx \frac{1}{L}\sum_{l=1}^L\ln p(Z^{(l)},X|\theta) \quad \text{[Eq.(4)]}"),
    eq_block(r"Q(\theta,\theta^{\mathrm{old}}) \approx \frac{1}{L}\sum_{l=1}^L\ln p(Z^{(l)},X,Y|\theta) \quad \text{[Eq.(6)]}"),
    para(rt("2022 論文は MCEM を採用し Q 関数をサンプル平均で近似している。")),
    h3("事後分布（2022 Eq.(9)(10)）— Y=Gaussian なので正確な Gaussian"),
    para(rt(
        "Y=Gaussian のため A_Y''(η)=1（定数）→ Hessian が z_i に依存しない。"
        "よって事後分布が正確な Gaussian。"
        "2022 の Ai は精度行列（精度行列の定義に ^{-1} なし）として Eq.(9) に定義される："
    )),
    eq_block(
        r"A_i^{[\text{2022}]} = \frac{1}{\sigma_z^2}I + \frac{1}{2\sigma_y^2}\sum_{j\neq i}z_jz_j^\top + F^\top\Sigma^{-1}F"
        r"\quad\text{[Eq.(9), 精度行列]}"
    ),
    eq_block(
        r"\eta_i = A_i^{-1}\!\left(F^\top\Sigma^{-1}(x_i - \mu) + \frac{1}{\sigma_y^2}\sum_{j\neq i}y_{ij}z_j\right)"
        r"\quad\text{[Eq.(10), MAP 推定値]}"
    ),
    callout(
        "記法注意：2022 のガウス事後分布は N(η_i, A_i^{-1})（共分散 = A_i の逆行列）。\n"
        "Woodbury 式（Eq.(17)-(20)）で A_i^{-1} を効率計算する。\n"
        "2024 の Ai（Eq.(22)）は共分散行列そのものとして定義されており、記法が逆である。",
        "💡", "blue_background"
    ),
    callout(
        "1/(2σ²_y) の由来：{∏_{i≠j} p(y_ij)}^{1/2} の 1/2 べきから来る（Eq.(5)）。\n"
        "すなわち 2022・2024・Dual-ExpFam は全て Y 側精度行列項に 1/2 因子が現れる。\n"
        "本提案の (w²/(2a_Y)) と対応（2022 では w=1, a_Y=σ²_y なので 1/(2σ²_y)）。",
        "💡", "blue_background"
    ),
    h3("M-step の閉形式解（2022 Eq.(12)-(16)）"),
    tbl([
        ["パラメータ","閉形式解（Eq. 番号）"],
        ["F","(Σ_l Σ_i (xi-µ) z_i^(l)T)(Σ_l Σ_i z_i^(l) z_i^(l)T)^{-1}  [Eq.(12)]"],
        ["µ","(1/Ln) Σ_l Σ_i (xi - Fz_i^(l))  [Eq.(13)]"],
        ["Σ","diag of (1/Ln) Σ_l Σ_i (xi-Fz_i-µ)(xi-Fz_i-µ)^T  [Eq.(14)]"],
        ["σ²_y","(1/Lnh) Σ_l Σ_{i,j≠i} (yij - z_i^T z_j)² / 2  [Eq.(15), h=C(n,2)]"],
        ["σ²_z","(1/Lkn) Σ_l Σ_i ||z_i||²  [Eq.(16)]"],
    ]),
    h3("Woodbury による高速化（2022 Eq.(17)-(20)）"),
    para(rt(
        "Ai^{-1} の直接計算は O(k³)。A = Ai + (1/(2σ²_y)) z_i z_i^T とおき [Eq.(17)]、"
        "Woodbury の公式から [Eq.(18)]："
    )),
    eq_block(
        r"A_i^{-1} = \left(A - \frac{z_iz_i^\top}{2\sigma_y^2}\right)^{-1}"
        r"= A^{-1} - A^{-1}z_i\frac{-2\sigma_y^2 + z_i^\top A^{-1}z_i}{1}\,z_i^\top A^{-1}"
    ),
    para(rt("A^{-1} を一度計算しておけば各 i の Ai^{-1} は O(k²) で更新できる（Eq.(19)(20)）。")),
]))

B.append(toggle("▶ 先行研究詳細 (2)：Mikawa 2024 [A_study_on...pdf] — Bernoulli モデルと Laplace 近似", [
    h3("生成モデル（2024 Eq.(1)(2)）"),
    eq_block(
        r"x_i = Fz_i + \mu_x + \varepsilon_i,\quad \varepsilon_i\sim\mathcal{N}(0,\Sigma)"
    ),
    eq_block(
        r"y_{ij}\sim\mathrm{Bern}(s_{ij}),\quad s_{ij}=\sigma(w_0+w z_i^\top z_j)"
    ),
    para(rt("2022 から y の分布を Gaussian → Bernoulli に変更。w_0, w を新たに導入。")),
    h3("尤度関数（2024 Eq.(16)）"),
    eq_block(
        r"\mathcal{L} = \prod_{i=1}^n p(x_i|F,z_i,\mu_x,\Sigma)"
        r"\left\{\prod_{i\neq j}p(y_{ij}|z_i,z_j,w_0,w)\right\}^{1/2}"
    ),
    h3("Q 関数（2024 Eq.(17)(18)）"),
    eq_block(r"Q(\theta;\theta^{\mathrm{old}}) = \int p(Z|X,Y;\theta^{\mathrm{old}})\ln p(Z,X,Y|\theta)\,dZ \quad\text{[Eq.(17)]}"),
    eq_block(
        r"Q \approx \frac{1}{L}\sum_{l=1}^L"
        r"\Bigl[\ln p(Z^{(l)}|\theta) + \ln p(X|Z^{(l)};\theta) + \ln p(Y|Z^{(l)};\theta)\Bigr]"
        r"\quad\text{[Eq.(18)]}"
    ),
    h3("事後分布の分子の対数（2024 Eq.(19)(20)）"),
    eq_block(
        r"\ln f(Z|X,Y;\theta)"
        r"= \sum_i\left[-\frac{\|z_i\|^2}{2\sigma_z^2}"
        r"- \frac{1}{2}(x_i-Fz_i-\mu_x)^\top\Sigma^{-1}(x_i-Fz_i-\mu_x)"
        r"+ \frac{1}{2}\sum_{j\neq i}\bigl\{y_{ij}\ln s_{ij}+(1-y_{ij})\ln(1-s_{ij})\bigr\}\right]"
        r"\quad\text{[Eq.(20)]}"
    ),
    h3("Laplace 近似（2024 Eq.(21)(22)）"),
    eq_block(r"p(z_i|x_i,y_{ij},Z_{\setminus i};\theta)\approx\mathcal{N}(\eta_i,A_i) \quad\text{[Eq.(21)]}"),
    eq_block(
        r"A_i = \left(\frac{1}{\sigma_z^2}I + F^\top\Sigma^{-1}F"
        r"+ \frac{w^2}{2}\sum_{j\neq i}s_{ij}(1-s_{ij})z_jz_j^\top\right)^{-1}"
        r"\quad\text{[Eq.(22), 共分散行列 = \Lambda_i^{-1}]}"
    ),
    para(rt("Ai は共分散行列（= Λ_i^{-1}）。η_i は Newton 法で MAP を求めて得る（2022 の Ai と定義が逆）。")),
    h3("gradient（2024 Eq.(23)）— 2022 Appendix-A より導出"),
    eq_block(
        r"\frac{\partial\ln f}{\partial z_i}"
        r"= -\frac{z_i}{\sigma_z^2} + F^\top\Sigma^{-1}(x_i-Fz_i-\mu_x)"
        r"+ \frac{w}{2}\sum_{j\neq i}(y_{ij}-s_{ij})z_j"
        r"\quad\text{[Eq.(23)]}"
    ),
    para(rt("Appendix A の導出：∂E/∂z_i = (1/2) Σ_{j≠i} (y_ij - s_ij) w z_j [Eq.(A-5)]。")),
    h3("w_0, w の gradient（2024 Eq.(24)(25)）"),
    eq_block(
        r"\frac{\partial\hat{Q}}{\partial w_0} = \frac{1}{L}\sum_l\sum_{i}\sum_{j\neq i}(y_{ij}-s_{ij})"
        r"\quad\text{[Eq.(24)]}"
    ),
    eq_block(
        r"\frac{\partial\hat{Q}}{\partial w} = \frac{1}{L}\sum_l\sum_{i}\sum_{j\neq i}(y_{ij}-s_{ij})z_i^\top z_j"
        r"\quad\text{[Eq.(25)]}"
    ),
    para(rt("注：2024 Eq.(24)(25) は j≠i の和（全ペア）。本稿 2.7 の i<j の和と等価（対称性による）。")),
    callout(
        "2024 の核心：Bernoulli の s_ij(1-s_ij) = A_Y''(η) が精度行列の Y 側に現れる。\n"
        "本提案はこれを A_Y''(η)/a_Y(φ_Y) に一般化し、任意の 1-パラメータ ExpFam に対応させた。",
        "💡", "blue_background"
    ),
]))
B.append(divider())
