# 監査: 細胞全体のセグメンテーションを導入する前の、既存コードと画像データ（2026-11-18）

新しいアルゴリズムを入れる前に、既存の研究コードとデータを確認した結果です（要件 9）。

## 1. 画像データ

| サンプル | 切片 | 種類 | 画素 µm | level 0 (H, W, C) | ピラミッド | 範囲 µm |
|---|---|---|---|---|---|---|
| 031-6_2 | H&E | brightfield RGB uint8 | 0.1887 | 11411×11243×3 | 5 段 | 2122×2153 |
| 031-6_2 | IF | **表示用 RGB uint8**（`fluorescence_display_rgb`） | 0.1002 | 10240×10240×3 | 5 段 | 1026×1026 |
| 031-8_6 | H&E | brightfield RGB uint8 | 0.1887 | 11400×11215×3 | 5 段 | 2117×2151 |
| 031-8_6 | IF | 表示用 RGB uint8 | 0.1002 | 10240×10240×3 | 5 段 | 1026×1026 |

- **IF には生の蛍光チャンネルがありません。** OME-TIFF の中身は 8 bit で合成された表示用 RGB（YXS）だけです。
- 提供元の assets.json に書かれた割り当ては次のとおりです。
  - R = Desmoplakin
  - G = Dystrophin1
  - B = Hoechst + Desmoplakin
- 核の推定には B−R（Hoechst の近似）を使いました。以前の研究では、既存の核に対する AUC が 0.89 でした。
- この画像では、マーカーの値が表示用の階調処理と飽和を経ています。チャンネル間の漏れ込みもあります。
- このため、Mesmer と Cellpose に入力した「膜」「核」は、すべて表示用 RGB から作った**代用チャンネル**です（REPORT §2）。
- 膜関連マーカー（Dystrophin は心筋細胞の筋鞘、Desmoplakin は介在板）が**あることと、細胞境界を正確に取り出せることは別問題**として扱いました。
  - Dystrophin は心筋細胞を囲みます。間質の細胞（線維芽細胞・内皮細胞など）は囲みません。
  - Desmoplakin は細胞の端（介在板）に点や線として出るだけです。
  - したがって、膜信号で輪郭を取れるのは主に心筋細胞です。間質の細胞は原理的に取りにくいと予想しました（結果もそうでした。REPORT §4）。
- **H&E での細胞境界の見え方:**
  - 心筋の細胞質はエオシンで一様に染まり、細胞の間は細い淡い隙間として見える程度です。
  - 031-8_6 の横断面では隙間が比較的見えます。031-6_2 は斜め〜縦断面が多く、境界はほとんど見えません。
  - 核（ヘマトキシリン）ははっきり見えます。
- 031-6_2 の IF には、水平の継ぎ目（タイルの境界の暗い線、y≈632 µm 付近）があります。

## 2. 既存のセグメンテーション（すべて核）

| 切片 | ファイル（manifest） | 内容 | ID |
|---|---|---|---|
| H&E | `he_nuclei.csv` + 同名 `.geojson` | StarDist（analysis/he-stardist-seqif-registration-20261019）の**核** | `point_id`（CSV の行） |
| IF | `*_seg_seqif_level0.geojson` | 提供元の**核**セグメンテーション（面積の中央値 約 59 µm²） | `segment_id` |
| （H&E 側） | `*_seg_he.geojson` | IF の分割を H&E へ写したもの。H&E から独立した分割ではないので使っていない | — |

- 細胞全体（cell / membrane / whole-cell）のセグメンテーションは、どのサンプルにもありませんでした。
- 核輪郭は `nuclear_contours.load_contours` が元の ID で中心点に結びます（11-18 の研究）。
  - 031-6_2 と 031-8_6 では、座標系の不一致はありません。
- 核数: 031-6_2 は H&E 1242 / IF 1091、031-8_6 は H&E 4800 / IF 1552。

## 3. 既存コードで変更してはいけないもの（baseline_manifest.json、HEAD 1a66073）

- **変更しない 42 ファイル**（sha256 を記録し、テストで照合）。対象は次のとおりです。
  - Workflow C/D
  - engine
  - Graph/UOT/B2
  - Density / Joint Flow
  - viewer / scene / imaging / catalog
  - ページ
  - サンプルの manifest
  - **核輪郭のモジュール `nuclear_contours.py`**
- **変更を許可したファイル**（研究タブのみ）:
  - `cell_review.py`: 任意の `cell` 記録、schema /3、注釈の列
  - `cell_review_ui.py`
  - `cell_review_viewer.py`
  - `cell_compare_viewer.py`
- **新規**: `src/registration_research/cell_segmentation.py`（結果を読むだけ）
- 次の 3 つは、変更前の HEAD のコードと数値が一致することを確かめました（tests_wcs）。
  - 自動推定（Graph/UOT/B2）
  - 変形場（RBF を含む）
  - H&E 画像の変形
- 研究タブは `CELLREG_CELL_REVIEW=1` のときだけ表示されます。細胞輪郭の表示は、その中でさらに既定オフ（「表示しない」）です。

## 4. 計算環境（cell-rg とは別）

| 環境 | 内容 | 状態 |
|---|---|---|
| `cellseg-cp` | Python 3.11、cellpose 4.2.1.1（BSD-3）、torch 2.15.0+cu132、RTX 6000 Ada | 動作。cpsam の重み 1.15 GB、sha256 `e1440429…04abe2` |
| `cellseg-mesmer` | Python 3.10、DeepCell 0.12.10、TensorFlow 2.8.4 | 導入は完了。**重みを取得できず未実行**（`DEEPCELL_ACCESS_TOKEN` 未設定）。この環境の TF 2.8 は CUDA 11 のライブラリが無く、GPU は使えない（CPU のみ） |

- 環境の作成手順: `envs/setup_envs.sh`（ログは logs/、コミット対象外）
- **ライセンス**:
  - Cellpose（コード・cpsam の重み）は BSD-3-Clause です。
  - DeepCell / Mesmer は Modified Apache 2.0（非商用）で、重みの取得には users.deepcell.org の登録トークンが必要です。
  - トークンの取得と規約への同意は利用者の判断に関わるため、こちらでは行っていません。

## 5. 監査の結論（方式を選ぶ前の想定）

1. IF の膜代用チャンネル（G+R）があれば、心筋細胞の輪郭は取れる可能性があります。間質の細胞は難しいと予想しました。
2. H&E では境界の信号が弱いので、汎用モデルでも核しか取れない可能性が高いと予想しました。
3. どの方式でも正解の輪郭はありません。
   - 評価は「核との関係」「画像の境界信号との一致（代理指標）」「目視」で行います。
   - 精度の主張はしません。
   - 手動評価の仕組み（manual_eval/）を用意しました。
