# Sample provenance（2026-10-20）

Registration Explorerとこの研究で使うサンプルの出所と座標の扱いです。機械可読の情報は `viewer_sample_manifest.json` にあります（`build_viewer_manifest.py` で、実在するファイルだけから作成）。

## 1. 031-6_2

| 項目 | 内容 |
|---|---|
| H&E画像 | `projects/031-6_2/031-6_2_HE_031-6_XY02.ome.tif`（031-6 XY02）。0.18872 µm/px、11243 × 11411 px、約2122 × 2153 µm |
| 蛍光画像 | `projects/031-6_2/031-6_2_seqif.ome.tif`。0.1002 µm/px、10240 × 10240 px、約1026 × 1026 µm。表示用RGB（R=Desmoplakin、G=Dystrophin1、B=Hoechst+Desmoplakin） |
| H&Eの核 | StarDist `2D_versatile_he`（analysis `he-stardist-seqif-registration-20261019`）、1,242個 |
| 蛍光の核 | 提供者のGeoJSON `031-6_2_seg_seqif_level0.geojson`、1,091個 |
| 切片の関係 | **別の切片として扱います（2026-10-20のユーザー訂正）。** |
| 参照変換 | 提供者による画像NCCの配置。ground truthではなく、独立でもありません |

**訂正について**：2026-10-19の `REPORT.md` と `input_audit.md` には、031-6_2を「同じ切片」と書いた箇所があります。2026-10-20にユーザーから訂正があり、**別の切片同士**として扱います。この訂正は過去の記述より優先します。過去のanalysisは上書き禁止のため、元のファイルはそのままにして、ここに記録します。

- 影響：6_2で≤3 µm 0.599の近接が得られたことは、「同じ細胞の再検出」ではなく、「隣接する切片で、核の配置が局所的に保たれている」ことを示します。細胞IDの対応は、もともと主張していません。
- 位置合わせの評価（near-neighbor近接率、偶然一致との比、画像の整合）は、切片の関係によらず使えます。

## 2. 031-8_6

| 項目 | 内容 |
|---|---|
| H&E画像 | `projects/031-8_6/031-8_6_HE_reference_031-7_XY06.ome.tif`。**031-7 XY06**（031-8にはH&Eがない）。0.18872 µm/px、11215 × 11400 px、約2117 × 2151 µm |
| 蛍光画像 | `projects/031-8_6/031-8_6_seqif.ome.tif`。0.1002 µm/px、10240 × 10240 px、約1026 × 1026 µm |
| H&Eの核 | StarDist（同上）、4,800個 |
| 蛍光の核 | 提供者のGeoJSON、1,552個 |
| 切片の関係 | 別の切片（031-7と031-8）。**切片間の距離はUNKNOWN** |
| 参照 | なし。提供者のメモは「REFERENCE ONLY - not registered to this field」。対応領域は注釈されていません |

## 3. serial_section_pair_20260904（Point-only）

| 項目 | 内容 |
|---|---|
| 核 | Workflow Cのrun `runs/20260904-161247-698fe6c9/input/` の `original_moving_points.npy`（H&E、1,842個、crop内のpx）と `original_fixed_points.npy`（蛍光、1,979個、µm） |
| 画像 | このviewerには登録していません（Point-onlyの確認用） |
| spacing | H&E側のpixel spacingは、この記録の中ではUNKNOWN。変換は「H&E px → 蛍光 µm」をまとめて表します |
| 変換 | 既存A2のR2（analysis `real-registration-incremental-value-20261015`、`cache/real_stages.json`）。研究artifactとしてのverdictは記録されていないため、UIでは **UNVERIFIED（PREVIEW ONLY）** と表示します |

## 4. 座標の規約（3サンプル共通）

- modalityごとのframe-local worldを使います：world = level-0のpixel index × pixel spacing（x右、y下）。
- **stage座標・crop原点はUNKNOWN**です。OMEにPlane Positionがないためです。frame内の(0,0)を使い、推測では補完しません。
- 画像の左上の角はworld −0.5 × spacingです。pixelの中心が整数indexにある、という規約です（核座標と同じ）。
- 撮像範囲は「画像サイズ × spacing」で求めます。点群のbounding boxは撮像範囲として使いません。
