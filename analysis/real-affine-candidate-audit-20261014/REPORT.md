# Real Affine / Coordinate / Candidate Discovery Audit

実施日: 2026-10-14。対象: `serial_section_pair_20260904`。本監査はanalysis専用であり、Workflow C/UI/D、Graph/UOT、safety gate、B2 calibration、deformation fieldを変更していない。

| Audit | Result | Evidence | Severity |
|---|---|---|---|
| Coordinate system | UNKNOWN | 保存配列の変換は誤差0で再現。一方、raw HE spacing/crop/downsamplingとGeoJSON CRS/unit metadataが未保存 | HIGH provenance gap |
| Affine direction | PASS | Moving→Fixed式を保存配列へ適用した最大誤差 0.0e+00 µm、往復 1.7e-13 px | LOW |
| Affine alignment | UNKNOWN | Affine後NN中央値 9.90 µmだが独立landmark/TREなし。代理指標だけでPASSにしない | MEDIUM |
| ROI overlap | UNKNOWN | 保存subset filterは完全再現、点群hull IoU proxy 0.993。蛍光取得範囲polygonなし | MEDIUM |
| Regional NCC | FAIL | ±18 µmで 1025/1754 (58.4%) が境界。±54でも 33.4% | HIGH |
| Candidate generation | UNKNOWN | 全1,754核で30候補・index整合。Real真値がなく、plausible≠correctを判定不能 | MEDIUM |
| Candidate pruning | UNKNOWN | union中央値 37→30。ただしK=10–100でraw Graph top-1は≥99.9%安定し、Realの正答除外は判定不能 | MEDIUM |

判定基準: PASSは実装経路・保存artifact・独立参照が整合する場合、FAILは再現可能な仕様違反または診断条件の明確な破綻、UNKNOWNは真値/authoritative metadataなしでは結論できない場合。幾何学代理指標だけでcell correspondenceをPASSにしていない。

## Executive finding

明確な逆行列誤用、Affine二重適用、subset index崩れは見つからなかった。最大の再現可能な前段問題は **D. REGIONAL NCC SEARCH LIMITATION** である。ただし境界到達は単純な「窓が18 µm不足」だけではなく、低い/平坦なNCC peakと遠方density peakへの吸引を含む。窓拡大はbest NCCを 0.360→0.734 に上げる一方、offset中央値は (-6, -45) µm、Graph support中央値は 0.244→0.048となり、Graph-cost top-1を 87.6% 変えた。真値なしでは改善と呼べない。

座標の内部計算は整合するが外部metadataが欠落するため **G. INCONCLUSIVE** も併記する。B. AFFINE MISALIGNMENTは局所proxyに示唆はあるが、独立landmarkがなく確定しない。A/C/E/Fはいずれも現時点で確定しない。

## Q1. Is the coordinate system correct?

内部経路は整合する。Raw HE normalized xy pixel → x flip (`901-x`) → `p @ A.T + t` → Fixed GeoJSON world xy である。有効scaleは 1.0965/1.1004 µm/input-pixel、effective determinantは -1.2066。保存Affine点は元1842点からexactに再構成できる。

ただしraw pixel spacing、crop、downsampling、GeoJSON CRS/unit宣言、入力`xy/yx`選択値が保存されていないため、物理座標系全体はUNKNOWN。詳細は [coordinate_system_audit.md](coordinate_system_audit.md)。

## Q2. Is the Affine transform applied correctly?

はい、保存artifactに対する実装監査ではPASS。保存行列は:

```text
[[-1.0964392   0.01346399]
 [-0.0081116  -1.10035744]]
translation=[1002.81039098 1013.49609085]
```

方向はMoving→Fixedで、再構成誤差0、逆変換往復誤差 1.71e-13 px。candidate codeは既にAffine後の`success_metric_moving_points.npy`を読み、再適用しない。

### Affine estimation source audit

- Input: HE 1,842核pixel xyをmoving、蛍光GeoJSON 1,979核world xyをfixedとする。画像値はobjectiveに使わず、901×901の寸法だけをflipに使う。
- Initialization: x/yの4 flip候補ごとに0–330°を30°刻みで試し、centroidとRMS半径からscale/translationを初期化。
- Optimization: nearest-neighbor trimmed similarity ICP（採用分位0.8）後、full 2×2 affine least-squares ICP（採用分位0.7）。各最大80 iterations、tol=1e-4。
- Flip選択: full-affine最終残差そのものではなく、similarity-stage orientation error + negative-determinant penalty + explicit-flip count penaltyを主scoreとし、最終median residualはtie-breakerにのみ使う。
- 同一入力で再推定すると保存matrix/translation/flipにbit-exact一致。採用pair=1289、trimmed mean residual=7.537 µm、全点final median NN residual=9.838 µm。

なおflip式は `width - x`（`width-1-x`ではない）だが、1 pixelの定数差は後続Affine translationへ吸収され、保存点のexact reconstructionを妨げない。

## Q3. Does Affine leave systematic spatial residuals?

Affine後NN距離中央値は 9.90 µm、mutual-NN率は 54.8%。全体NN vector中央値は (0.19, 0.41) µm、NCC offset中央値は (0.00, 3.00) µm。

格子regionで両vectorの方向cosine中央値は 0.422、低Graph support quartileでは 0.821。これは局所的な系統残差の可能性を示すが、NN/NCCは真の対応ベクトルではないためAffine misalignmentの証明ではない。独立landmarkがなくTREは算出していない。

局在性として、NCC offset中央値が最大のregionは `r5_c0`（中心 108.1, 921.9 µm、norm 23.4 µm）、NN距離中央値最大は `r5_c0`（15.0 µm）、Graph support中央値最低は `r5_c0`（0.045）。単一方向のglobal translationだけでなく、局所差が残る。

## Q4. Why do 58.4% of regional offsets reach the search boundary?

±18 µmで核 1025/1754、独立regionは 193/331 が境界に到達した。Regional offsetは核ごとの独立推定ではなく、42 µm bin内で3核以上の **331 region anchor**を作り、各核は最近傍anchorのoffsetを継承する。

核単位の方向別内訳は X+ 307、X− 310、Y+ 369、Y− 306、X/Y両方向 267。独立regionではそれぞれ58、58、68、59、50。X/Yおよび符号別countは重複し得る。

| ±18 group | Best NCC | Second NCC | Peak margin | Offset norm (µm) | Local density (/µm²) | Candidate distance (µm) | Graph support |
|---|---:|---:|---:|---:|---:|---:|---:|
| Boundary | 0.340 | 0.309 | 0.037 | 21.63 | 0.0019 | 17.45 | 0.206 |
| Non-boundary | 0.391 | 0.356 | 0.034 | 10.82 | 0.0019 | 11.06 | 0.301 |

窓を±36/±54へ広げると核の境界率は 54.0%/33.4%、独立regionでは 179/331、108/331。±54ではY offset中央値が -45 µmへ移りGraph supportが低下したため、一部の探索範囲不足と、識別力不足/遠方density peakの両方を支持する。best NCC上昇だけをregistration改善とは解釈しない。

## Q5. Can candidate generation be missing plausible correspondences?

可能性はあるがReal真値なしでは確定できない。実装は正確に:

1. exact spatial top-30。
2. `moving + regional offset`からregional residual top-30。
3. spatial top-80内でfingerprint top-30。
4. 各source top-20をstable-order union。
5. Full base costでtop-30へprune（不足時はcombined/spatialで補充）。
6. Graphは削除せずcostのみ変更。

source top-30 Jaccard中央値はspatial–regional 0.765、spatial–FP 0.250、regional–FP 0.250。Spatial top-1と最終Graph top-1は 52.7% で異なる。exact spatialの30番目距離中央値は 70.60 µm、80番目は 118.90 µm。

固定augmented poolでK=30→100にしてもGraph top-1 stabilityは 1.000、exact-spatial K=10でも 0.999だった。したがって今回のraw Graph-cost top-1については、K=30 pruningが主要因という証拠は弱い。ただしReal真値がないため、正候補を落としていないとは断定できない。Synthetic positive controlsでは正しいAffineからlarge translation errorへ、true-match candidate recall@30が 1.000→0.961となり、監査指標が既知Affine誤差に感度を持つことを確認した。

## Q6. What explains the low Real Graph support?

B2と同一定義のReal top-1 Graph support中央値は 0.227。観測事実は、低密度、±18 NCC境界率58.4%、source候補の低い重複、Affine後NN約9.9 µm、局所NN/NCC方向の不完全な一致である。これらはGraph近傍の対応候補が同時に整合しにくいことを部分的に説明する。

しかし、Real-like syntheticとの差すべてをAffine/ROI/candidateだけで説明できない。真値なし、source metadata不足、連続切片の生物学的差のため **F. BIOLOGICAL / DOMAIN DIFFERENCEは候補だが確定しない**。

## Q7. What should be fixed first?

今回は修正していない。次の順序を推奨する。

1. **入力provenanceを必須保存**: raw HE pixel spacing、crop/downsampling、axis/order、GeoJSON CRS/unit、元feature ID、UI filter値。これはアルゴリズム変更ではなく、現在UNKNOWNの座標監査を閉じる。
2. **独立landmarkを少数取得**し、Affine TREと方向を検証する。NN/NCC proxyでAffineを再最適化しない。
3. **NCC wideningを採用せず診断継続**: ±36/54のpeakをimage/manual landmarksまたはheld-out synthetic truthで検証し、遠方偽peakを区別する。その後にcandidate K/pruningのsynthetic Recall@Kを評価する。

## ROI / subset audit

Original→subsetはFixed 1979→1830、Moving 1842→1754。保存HE tissue mask、edge margin 10 px、Fixedのnearest-HE上限43 µmという既存UI defaultを用いると両subsetをexact再現した。点群convex-hull intersectionはFixed hullの 99.7%、Moving hullの 99.5%、IoU 0.993。ただしhullは真の撮像polygonではない。

## Synthetic positive-control summary

| Condition | Recall@30 | Spatial rank median | NCC boundary | Graph top-1 accuracy |
|---|---:|---:|---:|---:|
| compound_error | 0.996 | 8.00 | 11.1% | 0.487 |
| correct_affine | 1.000 | 1.00 | 7.4% | 0.741 |
| large_translation_error | 0.961 | 27.00 | 92.1% | 0.095 |
| rotation_error | 1.000 | 1.67 | 8.4% | 0.721 |
| scale_error | 1.000 | 2.00 | 7.0% | 0.689 |
| small_translation_error | 1.000 | 2.00 | 8.6% | 0.666 |

これは監査指標のpositive controlであり、Realの位置ずれを直接証明しない。

## Validation and reproducibility

- ±18 vectorized NCC vs frozen trace maximum NCC error: 2.22e-16。
- Fast sparse Graph vs frozen literal loop maximum error: 0.00e+00。
- Stable original-row IDsは全subsetで一意。元GeoJSON feature IDはartifactで失われたため、`fixed_original_######`を使用。
- NaN/Inf、candidate index、boundary condition、determinismをanalysis testsで検査。
- 読み取り前後のproduction source hash一致: True。
- Analysis-only tests: **11 passed**。Full repository regression: **269 passed, 4 skipped**。
- 監査runtime: 778.2 s。

## Interpretation limits

Realのgeometric plausibility、NN、NCC、Graph supportはtrue cell correspondence accuracyではない。candidate数30があることは正候補の存在を保証しない。窓/K拡大でscoreや候補数が増えても改善とはみなさない。本監査はmatching adoption、UOT mass、warpingを変更・評価していない。
