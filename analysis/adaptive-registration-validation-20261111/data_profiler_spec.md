# Data Profiler 仕様（研究用 v0.1、2026-11-11）

試作：`data_profiler.py`（`pair_profile()`）。研究モードだけで使う。本番コードからは import しない。
実測値は `profiles.csv`、`profiles.json` にある（今回のケース 16 件）。

## 1. 原則
- **探索前（pre-search）の指標**と**探索後（post-search）の指標**を、型として分ける。
  - Profiler が返すのは探索前の指標だけ。
  - 探索後の指標は Validation 層が返す。
- 探索前の指標は、片側の画像か核か、メタデータだけから計算する。候補の姿勢には依存しない。
  - そのため、探索や検証と循環しない。
- 切片の関係は、次の 3 つを別々の欄に記録する。
  - メタデータ（来歴）
  - ファイル名から推測した値（ヒント）
  - 探索後に観測した核の一致度
- 観測した核の一致度から `SAME` を推定することは、しない。
- `UNKNOWN` は `SAME` として扱わない。
- しきい値はすべて**暫定**。実データ 2 サンプルから決めたもので、較正していない（§4）。

## 2. 探索前に測れる指標（Profiler の出力）

| グループ | 指標 | 定義（実装） | 何の判断に使うか |
|---|---|---|---|
| 入力の有無 | `images_both`, `nuclei_both` | 画像と核のファイルが両側にあるか | 実行できる探索手段 |
| 座標系 | `coordinates_consistent` | 両側の pixel spacing が既知で、核の 95% 以上が撮像範囲の中にある | 不整合なら探索しない。メタデータの修正を先に依頼する |
| 座標系 | `pixel_um`, `extent_um`, `area_mm2` | manifest の値 × 画像サイズ | FOV の比 |
| FOV | `fov_area_ratio` | fixed の面積 ÷ moving の面積 | 大きいほど、候補が一意だという根拠が弱い |
| 画像の品質 | `image_usable` | engine の `features.extract` と同じ品質判定（組織の割合が 0.02〜0.995、自身の核で測った nuclear map の AUC > 0.7） | 画像探索を使えるか |
| 蛍光チャンネル | `occ_nuclear_corr` | occ map と nuclear map の相関（8 µm 格子） | 高いと「組織」チャンネルが実際には核を拾っている（10-27 に起きた失敗） |
| 組織 | `tissue_fraction`, `envelope_fraction`, `occ_mean` | tissue_bin の割合、外形の割合、占有率 | 組織の少ない視野 |
| 組織の外形 | `boundary_info` | 1 − envelope の割合（envelope ≥ 0.98 なら 0） | 視野に組織の外縁があるか（8_6 はほぼ全面が組織の中） |
| 間隙構造 | `cleft_strength` | 外形の内側で測った band-pass cleft の標準偏差 | occ+cleft 探索の情報量 |
| 反復性 | `repetitiveness` | cleft の自己相関の、80 µm を超えるずれでの最大値 | 似た構造が繰り返すと、誤った候補や象限の偶然一致が起きやすい |
| 核 | `n_nuclei`, `nuclei_per_mm2`, `nuclear_map_auc`, `nuclei_in_tissue_fraction` | 核の数・密度・品質 | 点群探索の情報量 |
| 核 | `nuclei_density_ratio` | moving と fixed の核密度の比 | 検出条件の違い（6_2 の HE は密度が低い） |
| 切片の関係 | `relation.metadata`, `relation.filename_hint`, `relation.confidence` | `CONFIRMED_METADATA` / `CONFLICT` / `FILENAME_HINT_ONLY` / `NONE` | 検証ゲートの選択（§3） |

## 3. 探索後にしか測れない指標（Validation 層。Profiler には入れない）

| 指標 | 出どころ（engine 1.2 で既に記録しているもの） |
|---|---|
| 候補の margin と一意性の判定（UNIQUE / AMBIGUOUS / LOW_PEAK） | `localization.<src>.verdict`, `margin` |
| 候補の重なり率、重なり領域の核の数 | `validation.overlap` |
| 核の近接率（≤3 µm）と mirror null | `validation.fixed_frame`, `if_frame` |
| ICP による移動量、`RELOCATED_BY_ICP` | `refine.pose_change`, `flags` |
| 象限一致（局所） | `validation.quadrant_consistency`（隣接切片の image#0 だけ） |
| 探索手段どうしの一致（image#0 と points#0 の距離） | 未記録。collect.py で計算した（`image_vs_points_top_um`） |
| 象限ごとの全域探索の一致 | 未実装。研究用に `quadrant_global.py` を試作 |
| 変換の妥当性（scale、異方性、reflect） | `T_final`。妥当性の判定はまだない |

## 4. 実測値と暫定しきい値（`profiles.csv`）

| ケース | IF の cleft の強さ | occ と核の相関 | 外形の情報 | 反復性 | FOV の比 | 切片関係の信頼度 |
|---|---|---|---|---|---|---|
| 8_6（元のまま） | 0.284 | −0.20 | 0.15 | 0.065 | 4.3 | CONFIRMED_METADATA |
| 8_6 IF500 | 0.277 | −0.23 | 0.00 | 0.073 | 18.2 | 同上 |
| 6_2（元のまま） | 0.241 | 0.29 | 0.59 | 0.109 | 4.3 | **CONFLICT** |
| 6_2 IF500 | 0.204 | 0.08 | 0.20 | 0.171 | 18.3 | **CONFLICT** |

- この 2 サンプルは、特徴の範囲が狭い。間隙は両方ともはっきりしていて、反復性は両方とも低い。
- そのため、しきい値の**下限側**（構造が弱い場合、反復が強い場合）は、データで確かめられていない。
- `adaptive_strategy.DEFAULT_RULES` の値は、暫定のプレースホルダーとして置いたもの。

## 5. 実データが届いたときの手順
1. `viewer_sample_manifest.json` に新しいサンプルを追加する。
   - pixel_um、画像のパス、核のパスを入れる。
   - `provenance.same_section` には来歴から分かる値を入れる。不明なら null。
2. `prep.py features` で特徴 map をキャッシュする。
3. `cases.json` にケースを追加する（ops は空、reference は null か、ランドマークから求めた変換）。
4. `data_profiler.py` を実行する。
5. `benchmark_runner.py` を Slurm の array で実行し、`collect.py` で集計する。
