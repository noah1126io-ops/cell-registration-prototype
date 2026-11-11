# 独立Synthetic generator（`indep_gen.py`）

**位置づけ**：組織を模した人工的な2D点過程。実際の病理組織を再現したものではない。3Dの核楕円体や切片の厚さはモデル化していない（2Dの部分共有モデル）。

## 前回generatorとの独立性

前回の `init_lib.make_pair` とは、次の点で独立している。

- **コード**：`make_pair`、`_intensity`、`_sample` を呼ばない（testで確認）。
- **潜在密度**：前回は「Gaussian bump 10個 + 矩形の裂隙」で作っていた。今回は族ごとに異なる過程を新たに実装した（下記）。
- **座標系**：前回は2400 µmの固定組織を使った。今回は両FOVの外接矩形＋120 µmを潜在領域とする。
- **変形**：前回は正弦波の変形を使った。今回はGaussian random fieldを使い、affine成分を除去する。
- **Seed**：`SHA-256(purpose|family|condition|rep)` から派生させる（`SEED_ROOT=20261015`）。
  - 前回は11–13、101–105、301、401、2101–4105などのseedを使っていた。
  - 開発中に、key文字列が4 byteに切り詰められて衝突するbugを発見し、hash方式に修正した。A2を一度も実行する前のことで、修正は回帰testで保護している。
- **真値の扱い**：評価用の真値や前回の成功結果は、点群の生成に一切使っていない。

## 族（テンプレート → 核サンプル）

すべての族で hard-core 5 µm（ランダム順の逐次抑制）を適用する。過剰に生成された分は、一様に間引いて目標数に合わせる。

| 族 | テンプレート | 核サンプル |
|---|---|---|
| G1 Clustered | Thomas過程の親点（密度κ）。子の散らばりσ ~ U(14, 32) µm、平均子数μ ~ U(8, 16) | 親ごとにPoisson(μ)個の子をN(0, σ²)で配置し、一様背景（目標数の40%）を加える |
| G2 Inhomogeneous | log-Gaussian場（相関長60–120 µm）× Voronoi 6–10領域の段差倍率（0.25–2.5倍、1領域は0.04＝急な密度低下）× 線形勾配 | 非一様Poisson（thinning） |
| G3 Structured Tissue | 上皮様の帯3–5本（正弦曲線、2–3列、核間隔7.5 µm）、腺管様の環（楕円の長半径35–80 µm、内腔は無核）、間質（疎な一様分布）、裂隙3–6本（幅15–40 µm、曲がり）、無核領域1–2個（半径70–180 µm） | 構造に沿ってjitter 1.4–1.6 µmで配置し、裂隙・無核領域・内腔の核を除去 |
| G4 Repeated / Ambiguous | 周期P ~ U(115, 170) µmのせん断・回転格子。motif = 環 + クラスタ + 一様点 | 格子のコピーごとに核を保持率0.8で残し、jitter U(2.5, 4) µmを加える。非周期の点を15%追加 |

- G4の密度自己相関は、格子ベクトルで0.30〜0.50、同じ半径の他方向では約0。つまり、探索範囲±200 µm内に**周期的な相関peak**が存在する。
- NC3は、G4のjitterを1.2 µm、保持率を0.92にした「ほぼ同一の繰り返し」である。

## 連続切片の生成過程（それぞれ独立した要因）

1. **共通核の割合**：潜在核ごとに {両切片: q_shared, fixedのみ: (1−q)/2, movingのみ: (1−q)/2} を割り当てる。
2. **検出漏れ**：切片ごとに独立なBernoulli（det_fixed, det_moving）。
   - 潜在密度は「観測fixed密度が目標値になる」ように補正する。
   - C07（q 0.30）とC08（q 0.85, det 0.5/0.5）は、観測上の共有率が同程度（約0.4）になるが、生成過程が異なる。
3. **局所的な核集団の消失**：各切片に半径40–110 µmの円patchをk個置き、その中の核を95%の確率で除去する。
4. **2D重心の測定誤差**：切片ごとに独立なN(0, 0.6² µm²)。
5. **滑らかな局所変形**（moving切片のみ）：
   - 相関長110–200 µmのGaussian random fieldを使う。
   - moving FOV内での最小二乗affine成分を除去し、RMSを指定値に合わせる。
   - これにより、真のglobal Affineと切片固有の局所変形が分離される（testで、残るaffine係数がほぼ0であることを確認）。
6. **撮像範囲**：
   - fixedは矩形FOV（µm）。
   - movingは、HE pixel画像（W×H px）をglobal Affineでworldに写したものとする：`world = M·p + t`、`M = pixel_um·R(θ)·C`（C = 反転）。
   - moving視野の中心は、fixed中心からoffset（方向はランダム）の位置に置く。

## 出力形式（前回の評価コードと共通）

`SectionPair` に次を保持する。

- `fixed`（µm）
- `moving_px`（px）
- `width`, `height`
- `M_true`, `t_true`
- `fixed_ids`, `moving_ids`（persistent ID）
- `true_fixed_index`（IDから導出）
- `moving_world_true`
- FOV多角形
- params

`to_init_pair()` で前回の `init_lib.Pair` に変換できる。

**重なり指標**：FOV多角形の交差面積（Sutherland–Hodgman）を使う。

- `overlap_frac_fixed`、`overlap_frac_moving`、`overlap_iou`
- `overlap_min_coverage` = 交差面積 / 大きい方のFOV面積（主指標）
- `centroid_shift_um`：点群重心間の距離。A2の並進探索の起点はこれで決まる。
