# MovieLens co-like recommendation 実験：Notion用整理サマリー

## 1. このページで伝えたいこと

MovieLens実験では、映画ジャンルXと共高評価人数Yを用いて、Bernoulli X / Poisson Y の実データ適用を確認した。また、推定された潜在変数Zの一部が、人気度・高評価率・年代・ジャンル群と関連している可能性を探索的に確認した。

## 2. なぜMovieLensを使うのか

- Wine実験: X=化学成分（連続値）, Y=同じカテゴリかどうか（Bernoulli）
- MovieLens実験: X=ジャンル0/1（Bernoulli）, Y=両方高評価した人数（カウント）
- MovieLensでは、Yを単純な「関係の有無」ではなく**Poissonのカウント関係**として扱える点がWine実験と異なる新しい検証ポイントになっている。

## 3. データ設計

- node = movie
- X = genre multi-hot, family_x = Bernoulli
- Y_colike_count[i,j] = 映画iとjを両方 rating >= 4 で評価したユーザー数
- family_y = Poisson
- subset = genre_stratified_mp100, n = 100, d = 19（MovieLens 100k全体のユーザー数 n_users=943）
- Y_colike_count: mean=14.75, max=99, density(>0)=0.993
- 補助実験: Y_lift_binary[i,j] = 1 if count>=min_support and lift>=threshold
  （lift = observed / expected, expected = like_count_i × like_count_j / n_users）
  選択値: min_support=20, lift_threshold=3.0 （density=0.067, positive_edges=332）

## 4. 主実験：Poisson co-like count

本文で見る指標はRMSE_Y・Pearson・BICの3つに絞る。AP/AUC/NDCG/MAPなどの詳細指標は補助実験・付録扱いとする。

**この実験はin-sample再構成であり、未知ペアの共高評価人数を予測できたわけではない。**


| k | RMSE_Y_mean | Pearson_mean | BIC_mean | short_interpretation |
| --- | --- | --- | --- | --- |
| 2 | 4.835 | 0.907 | 31809.5 | 次元が少なくシンプルだが、再構成誤差(RMSE_Y)は4条件中で最大 |
| 3 | 3.959 | 0.939 | 29435.1 | ジャンル構造との対応が比較的よい |
| 5 | 3.349 | 0.956 | 28667.3 | BIC最小でバランスがよい |
| 8 | 3.107 | 0.963 | 28812.1 | 再構成性能(RMSE_Y/Pearson)は最良だが複雑（BICはk=5よりやや悪化） |

## 5. Kの解釈

best_k_for_interpretation = 8（representative trial=2）。全8因子のうち、特に解釈しやすい3因子のみを示す。


| factor | tentative_label | evidence_correlation | top_high_movies_short | top_low_movies_short |
| --- | --- | --- | --- | --- |
| 2 | classic / well-regarded films (older release_year, higher avg_rating) | release_year r=-0.52; avg_rating r=0.49; high_rating_rate r=0.47 | Chinatown (1974); Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963); Young Frankenstein (1974) | Midnight in the Garden of Good and Evil (1997); I Know What You Did Last Summer (1997); Spawn (1997) |
| 4 | popularity-related (high like_count) | log_like_count r=0.75; like_count r=0.73; high_rating_rate r=0.65 | Good Will Hunting (1997); Apt Pupil (1998); Boogie Nights (1997) | Spawn (1997); Star Trek: The Motion Picture (1979); Alien 3 (1992) |
| 5 | high-rating / acclaimed-classics related (high high_rating_rate) | high_rating_rate r=0.78; avg_rating r=0.77; like_count r=0.70 | Lone Star (1996); Chinatown (1974); Postino, Il (1994) | Interview with the Vampire (1994); Craft, The (1996); Nightmare on Elm Street, A (1984) |

**注意点：潜在空間には回転不定性があるため、factorの意味は確定ではない。**

## 6. 補助実験：lift ranking

目的：人気度補正後の強い共高評価ペアを、単純な人気度・ジャンル類似度より上位に識別できるかを確認する。

item-itemベースラインは評価設計上の床効果（test positiveをtrain score上で0にする必要があるため）があり、公平な比較として説明が難しいため本文からは外す。


| method | AP_sampled | NDCG_at_10 | short_interpretation |
| --- | --- | --- | --- |
| popularity | 0.2 | 0.04 | 人気度だけではlift関係をほとんど説明できない（ランダムよりわずかに上） |
| genre_cosine | 0.265 | 0.138 | ベースライン中で最強だが、提案手法の半分以下 |
| popularity_genre | 0.24 | 0.173 | 人気度+ジャンルでも提案手法に届かない |
| proposed_dual_expfam | 0.677 | 0.408 | 全ベースラインに対しAP_sampled/NDCG@10とも最も高い値を示した |

## 7. 推薦例

**query: Citizen Kane (1941)**

- 提案手法Top5: Vertigo (1958) > North by Northwest (1959) > Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) > Taxi Driver (1976) > Lawrence of Arabia (1962)
- genre cosine Top5: Boogie Nights (1997) > Good Will Hunting (1997) > Gandhi (1982) > Taxi Driver (1976) > Emma (1996)
- コメント: 提案手法はVertigoやNorth by Northwestなど、ジャンルラベルだけでは出てこない強い共高評価関係を持つ古典名作を高く評価している。genre_cosineは同じDramaタグの映画（Boogie Nights, Good Will Huntingなど）を返すのみで、ジャンルを超えた構造を捉えていない。

**query: Vertigo (1958)**

- 提案手法Top5: Citizen Kane (1941) > North by Northwest (1959) > Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) > Taxi Driver (1976) > Lawrence of Arabia (1962)
- genre cosine Top5: Basic Instinct (1992) > Absolute Power (1997) > Client, The (1994) > Chinatown (1974) > I Know What You Did Last Summer (1997)
- コメント: 提案手法はCitizen Kaneなど質の近い名作群を返す一方、genre_cosineはMystery/Thrillerタグを持つだけの映画（Basic Instinctなど）を返し、必ずしも質的な近さを反映していない。

**query: Star Trek IV: The Voyage Home (1986)**

- 提案手法Top5: Clear and Present Danger (1994) > Star Trek VI: The Undiscovered Country (1991) > In the Line of Fire (1993) > Young Frankenstein (1974) > Highlander (1986)
- genre cosine Top5: Stargate (1994) > Star Trek VI: The Undiscovered Country (1991) > Star Trek III: The Search for Spock (1984) > Star Trek: Generations (1994) > Star Trek: The Motion Picture (1979)
- コメント: genre_cosineは同シリーズ作品（Star Trek系列）を機械的に返すのに対し、提案手法はシリーズ外の映画（Clear and Present Dangerなど）も高くランクしており、ジャンルラベルを超えた共高評価構造を示す一方、Star Trek VIのような直接の関連作も依然上位に残る。

**query: Mary Poppins (1964)**

- 提案手法Top5: Young Frankenstein (1974) > Vertigo (1958) > Gandhi (1982) > Glory (1989) > Citizen Kane (1941)
- genre cosine Top5: This Is Spinal Tap (1984) > Everyone Says I Love You (1996) > Grease (1978) > Mrs. Doubtfire (1993) > Mighty Aphrodite (1995)
- コメント: 【限界例】提案手法はVertigoやGandhi、Citizen Kaneのような全く異なるジャンルの名作ドラマを上位推薦しており、子供向けミュージカルとしての直感とは一致しにくい。人気度・高評価率に関連する因子がジャンルを問わず強く効いている可能性を示す例で、解釈には注意が必要。

## 8. 今回言えること

- Bernoulli X / Poisson Y の実データ適用例を作れた
- 共高評価人数をPoisson関係としてin-sampleで再構成できた
- BICではk=5がバランス良かった
- Kの一部因子は、人気度・高評価率・年代・ジャンル群と関連する可能性が見られた
- liftで定義した強い共高評価ペアについて、提案手法はpopularity/genre baselineより高いランキング性能を示した

## 9. まだ言えないこと

- ユーザー個人への映画推薦ができたわけではない
- 商用推薦システムとして使えるとは言えない
- Kの意味が完全に同定されたわけではない
- Poisson側はin-sample再構成であり、strict held-outではない
- lift rankingもzero-filled edge hidingであり、strict missing-pair CVではない
- n=100 subsetの結果であり、MovieLens全体の結論ではない
- item-item baselineとの公平な比較にはpair mask対応が必要

## 10. 次にやるべきこと

1. K解釈はrepresentative fitに基づくため、複数seedでの安定性確認が必要
2. MovieLens n=200/300への拡大
3. pair mask対応によるstrict held-out
4. 公平な推薦ベースライン比較
5. Negative Binomial Y
