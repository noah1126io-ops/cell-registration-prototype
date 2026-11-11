# Coordinate system audit (Claude, independent)

対象run: `runs/20260904-161247-698fe6c9`（`sweeps/real-grid27-20260908-v2/input` と全9ファイルSHA-256一致）。

## 判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| 変換経路の実装 | **PASS** | 保存Affine点 = flip→Affine(raw) を誤差0 µmで再構成、逆変換往復 1.7e-13 px |
| 変換方向 | **PASS** | HE pixel → 蛍光GeoJSON world。候補探索は保存済みAffine点を直接使い、二重適用なし |
| 軸・反転（chirality） | **PASS** | 反射（effective det −1.207）を含む現行向きでのみ局所correlogram peakが出る。mirror null（x/y/180°）はpeak比 3.3–3.5で背景水準 |
| 単位・scale | **PASS（proxy）** | 再fit後も ≈1.107 µm/px（保存1.10）。異方性0.07%。真のpixel spacing metadataは無い |
| ID/index | **PASS** | subset行は元配列へ一意・exact・単調に対応（fixed 1,830/1,979, moving 1,754/1,842） |
| Subset filter | **PASS（再現）** | 保存HE mask・edge margin 10 px・near-HE ≤43 µmで両subsetをexact再現 |
| 原点・crop・downsampling・CRS | **UNKNOWN** | metadata未保存（`point_io.py`/`geojson_utils.py`のTODO）。ただし下記Affine問題の原因候補として重要 |

**結論:** 座標規約の誤り（A）は検出されなかった。問題は規約ではなく、規約の上で推定された **Affineの位置（平行移動≈50 µm＋回転≈1.24°）** にある（`REPORT.md` Q2/Q3）。

## 変換経路（コード根拠）

1. `src/point_io.load_npy_centers`: HE核中心をpixel `(x, y)`に正規化（範囲 1.06–900.3 px、画像 901×901）。
2. `src/geojson_utils.load_geojson_centroids`: 蛍光核の`centroid_x/y`（なければ外周頂点平均）。CRS/単位は読まない。
3. `src/pointset_registration.estimate_affine_with_y_flip`: 4 flip × 12初期回転で similarity ICP（trim 0.8）→ affine ICP（trim 0.7）。初期化は **centroid一致＋RMS半径scale**。
4. 保存: `flip_x=True`（`x' = 901 − x`）、`world = p' @ Aᵀ + t`。
   - post-flip: det 1.2066, 回転 −179.44°, 特異値 1.1018/1.0951
   - raw→world実効: det −1.2066（y反転＝image y-down → world y-up）, 回転 +0.56°
5. `warp_he_image_to_world`: 出力 1.0 µm/px, lower-left原点。独立に `col=(x−col0)/1, row=(row0−y)/1` を導出し、HE核位置のHE画素輝度中央値 96 vs 組織内ランダム 144 → 画像と点は同じ変換で整合。
6. Workflow C subset: warped HE mask上で valid（edge margin 10 px, fixedはHE核から43 µm以内）。fixed除外: footprint外77, edge margin内72。moving除外: edge margin内83, footprint外5。
7. Contextual candidate / NCC / Graph / UOT / B2: `success_metric_*` 配列を入力。Affineの再適用なし。

## 副次的発見: "tissue mask" はtissueではない

`_tissue_mask_from_image(gray > 0.05)` はH&Eの明るいガラス・裂隙を背景として除外しない。保存maskは**HE画像のnonblack footprintと100%一致**（canvasの84.8%）。footprint内の7.7%はHE上の白色（無組織）領域。したがってsubset filterは「組織内」ではなく「HE画像範囲内」を意味する。

## 未保存metadata

raw HE pixel spacing、crop offset、downsampling、`xy/yx`選択値、GeoJSON CRS/unit、元feature ID。いずれも推定結果と矛盾する証拠はないが、HE視野と蛍光視野の約50 µmのずれ（下記ROI）を事前に検出するには、少なくとも両画像の取得範囲が必要。
