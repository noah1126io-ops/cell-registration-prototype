# Optional Image-Assisted Registration：画像を任意に使う連続切片位置合わせ（2026-10-19）

- 対象：両画像付きの新サンプル **031-6_2**（H&Eと蛍光が同じ切片）、**031-8_6**（H&Eは031-7で別の切片）
- 実装：analysis専用のprototype `oir/`。Productionは無変更です（設計は `architecture.md`、入力は `input_audit.md`）
- 計算：Slurm `mib-dbia`（CPU）。実データは21設定×2サンプル、半合成は32件×5手法、P0互換性の確認

## 0. 比較表

主設定はHE Fixed（E0のみ既存規約どおりHE Moving）です。E1はHE画像のみ、E3は結合規則 `zsum` です。蛍光画像を見せる、IF Fixedにする、などの変種は §2 にあります。

**031-6_2（同じ切片。参照は提供者のNCC変換で、独立ではない）**

| Method | Inputs | Region localization | NN median | ≤3 µm proximity | Image alignment（tissue Dice / 核map相関） | Runtime |
|---|---|---|---|---|---|---|
| E0 Point-only（既存A2） | Nuclear points | 失敗（参照から450 µm、scale 0.52） | 11.0 µm | 0.098（null 0.072） | 0.60 / 0.11 | 47 s |
| E0x Point-only（視野の大きさの違いに対応） | Nuclear points | **UNIQUE**（margin 0.48、参照から2.2 µm） | **1.91 µm** | **0.599**（null 0.049） | 0.90 / 0.80 | 396 s |
| E1 Single-image | Points + HE image | UNIQUE（E0xと同じ解） | 1.91 µm | 0.599 | 0.90 / 0.80 | 769 s |
| E2 Dual-image | Points + two images | LOW_PEAK（位置は正しい、margin 0.12）→ **POINT_ONLY_FALLBACK** | 1.91 µm（E0xを採用） | 0.599（E0xを採用） | 0.90† / 0.80 | 383 s |
| E3 Joint | Points + two images | UNIQUE（margin 0.53、E0xと同じ解） | 1.91 µm | 0.599 | 0.90† / 0.80 | 746 s |

**031-8_6（別の切片。正解なし）**

| Method | Inputs | Region localization | NN median | ≤3 µm proximity | Image alignment | Runtime |
|---|---|---|---|---|---|---|
| E0 Point-only（既存A2） | Nuclear points | 不採択（scale 0.43に崩れる） | 5.8 µm | 0.189（null 0.152） | 0.50 / 0.02 | 92 s |
| E0x Point-only | Nuclear points | AMBIGUOUS（margin 0.03） | 13.1 µm | 0.041（null 0.031） | 0.45 / 0.04 | 370 s |
| E1 Single-image | Points + HE image | AMBIGUOUS | 13.7 µm | 0.032 | 0.50 / 0.02 | 715 s |
| E2 Dual-image | Points + two images | AMBIGUOUS（margin 0.07） | 14.3 µm | 0.033 | 0.56† / 0.04 | 341 s |
| E3 Joint | Points + two images | AMBIGUOUS（margin 0.01） | 13.2 µm | 0.043 | 0.50† / 0.02 | 804 s |

- †：tissue mapを探索に使っているので、tissue Diceは独立した成功の根拠になりません（`img_tissue_independent_of_search=False`）。核mapの相関は、E2/E3の主設定では探索に使っていないので独立です。
- 031-8_6では、どの手法の変換も採択されていません（E0xへのfallbackも不採択で、「no transform accepted」と記録しています）。8_6の数値は、採択されなかった最上位の候補を評価したものです。
- 「null」は、鏡像にした点群で同じ評価をしたときの偶然の一致率です。

## 1. 主な結果

### 1.1 P0互換性（Phase B）

`cache/p0_compat.json`：

- 既存の実データ（前回analysisのR2）を、今回のwrapperから再実行しました。保存済みの結果との差は**行列0.0、並進0.0**です。
- 新サンプルでも、wrapper経由と直接呼び出しの差、画像を一緒に渡したときの差は、どれも0.0でした。
- **E0（既存A2）は変更していません。** 新サンプルで失敗するのは既存の仕様によるものです。A2は小さい視野のcanvasに大きい視野を合わせるので、2.1 mmのH&Eが1.0 mmの蛍光に切り詰められ、scaleが崩れます。

### 1.2 視野の大きさの違い（E0x）

- 点群だけで、canvasを両視野の和集合サイズに広げ、ICPを相手の撮像範囲内の点に限りました（E0x）。これで6_2は正しく位置合わせできます。
  - ≤3 µm：0.599（null 0.049の12倍）
  - Held-out ≤3 µm：0.45（HE Fixed）／0.59（IF Fixed）、null 0.016
  - 参照との差：中央値2.2 µm、p95 4.3 µm
  - 変換：scale 1.018、回転 −0.22°、鏡像なし、異方性 1.005
  - ICPによる変化：scale −0.3%、回転0.22°、点の移動3.5 µm
- 画像を探索に使わなくても、画像の整合性はよくなります（tissue Dice 0.90、境界のchamfer 6.3 µm、核mapの相関0.80、block相関の中央値0.73）。これは**独立した検証**です。
- 参照（提供者のNCC）との差は2.2 µmあります。参照は回転0固定のscale格子なので、どちらがより正しいかはこのデータでは決められません。

### 1.3 画像を追加したときの効果（E1〜E3、6_2）

| 設定 | 探索に使った層 | verdict / margin | 探索poseの≤3 µm | 最終結果 |
|---|---|---|---|---|
| E0x | points | UNIQUE / 0.48 | 0.51 | 基準 |
| E1he（HE画像のみ） | points + cross | UNIQUE / 0.27 | 0.51 | E0xと同じ |
| E1if（IF画像のみ） | points + cross | UNIQUE / 0.57 | 0.51 | E0xと同じ |
| E2（tissue） | img_tissue | LOW_PEAK / 0.12 | 0.30 | fallback → E0x |
| E2（IF Fixed） | img_tissue | AMBIGUOUS / 0.10 | 0.15 | fallback → E0x |
| E2 feat4（4 µmの特徴量） | img_tissue | UNIQUE / 0.11 | 0.21 | E0xと同じ |
| E2 c16f8（粗い探索） | img_tissue | LOW_PEAK / 0.12 | 0.51 | fallback → E0x |
| E2nuc（核map） | img_nuclear | **UNIQUE / 0.59** | 0.30 | E0xと同じ |
| E2tn（tissue＋核map） | img_tissue + img_nuclear | UNIQUE / 0.50 | 0.30 | E0xと同じ |
| E3 zsum | img_tissue + points | UNIQUE / 0.53 | 0.51 | E0xと同じ |
| E3 zmin | img_tissue + points | LOW_PEAK / 0.42 | 0.30 | fallback → E0x |
| E3 rerank | img_tissue → pointsで再評価 | UNIQUE / 0.62 | 0.30 | E0xと同じ |

- 採択されたものは、ICPの後ですべて同じ解（NN median 1.91 µm）に収束しました。**画像を足しても、最終的な精度は上がりません。**
- 画像だけ（E2）でも、6_2の正しい領域は見つかります。tissue mapだけではpeakが幅広く、事前に決めた規則（z < 6でLOW_PEAK、margin < 0.10でAMBIGUOUS）では採択されません。核mapを加えるとUNIQUEになります。これは**点群とは独立した領域推定**として、確認に使えます。
- 結合規則では、z の単純な和（zsum）と再評価（rerank）はmarginを保ちました。zminは弱い層に引きずられ、LOW_PEAKになりました。

### 1.4 031-8_6（別の切片）

- 点群・画像・結合のどの探索でもAMBIGUOUS（margin 0.006〜0.09）で、核配置の整合（≤3 µm ≥ 0.25 かつ null比 ≥ 3）にも届きません。
- HEが別の切片（031-7）なので、核1つ1つの対応はもともと期待できません。また、組織は密で、似た構造が繰り返しています。
- E2のHE Fixed／IF Fixedは4.3 µm差で同じ候補を選びました。ただしmarginが0.07〜0.08と小さく、核配置の整合もない（≤3 µm 0.033、null並み）ため、**対応領域は見つかっていない**と判断します。

### 1.5 陰性対照

- **実データ**：蛍光側の核をタイルごとに並べ替えた対照（NCshuf）。6_2・8_6とも、E0x・E2・E3のすべてで不採択でした。E2/E3ではICPが探索のtrust regionを外れ、棄却されています。
- **半合成**：真の領域を取り除いた8件（absent）。どの手法も誤って採択しませんでした（false accept 0/40）。E2では1件の探索がUNIQUEになりましたが、核配置の整合チェックで止まりました。

### 1.6 半合成（既知の変換・領域・細胞ID）

実際のH&Eから約1 mmの疑似蛍光windowを既知のsimilarity変換で切り出し、共有される核の割合qを変えました（各条件8件：6_2由来4件、8_6由来4件）。`synthetic_summary.csv`：

誤差と細胞ID一致率は、採択されたかどうかに関係なく、最終的な変換で8件それぞれを計算した値の中央値です。

| 手法 | 正しく採択（q 0.8 / 0.5 / 0.3） | 誤差中央値（q 0.8 / 0.5 / 0.3） | 真の細胞ID一致率（q 0.8 / 0.5 / 0.3） | 誤採択 |
|---|---|---|---|---|
| E0 | 0.63 / 0.88 / 0.38 | 0.13 / 0.31 / 253 µm | 1.00 / 1.00 / 0.00 | 0 |
| E0x | **1.00 / 1.00 / 1.00** | 0.08 / 0.28 / 1.10 µm | 1.00 / 1.00 / 0.98 | 0 |
| E1 | 1.00 / 1.00 / 1.00 | 0.08 / 0.28 / 1.08 µm | 1.00 / 1.00 / 0.98 | 0 |
| E2 | 0.50 / 0.50 / 0.50 | 0.12 / 412 / 1.11 µm | 1.00 / 0.50 / 0.97 | 0 |
| E3 | 1.00 / 1.00 / 1.00 | 0.08 / 0.28 / 1.10 µm | 1.00 / 1.00 / 0.98 | 0 |

- E2は由来で結果がはっきり分かれました。8_6由来では12/12で正しく採択、6_2由来では0/12です（AMBIGUOUS / LOW_PEAK）。6_2のH&Eは組織が視野の18%しかなく、まばらで、tissue mapの形だけでは位置を絞り込めません。
- ただし、生成器ではtissue mapの摂動が実際のモダリティの違いより単純です。逆に、疑似蛍光の核mapは点から描き直しています。そのため、半合成のE2の成績は実データの見積もりとしては粗く、上の数字は「どの条件でE2が崩れるか」の目安として扱ってください。
- 真の細胞ID一致率は半合成でだけ計算しました。実データでは、NNが近いことを細胞の同一性とはみなしていません。

### 1.7 計算コスト

`runtime_metrics.csv`（4 CPU）：

- 特徴量の抽出：2 µm gridで1 sampleあたり1秒未満（HE 0.7 s、IF 0.14 s）。pyramidのlevel 2〜3から読むので、level 0の画像全体は読みません。
- 探索：540 pose（chirality × 回転 × scale）で、E0x / E2は240〜510 s、ピークメモリ280〜370 MBです。
  - E1 / E3は層が2つになるので、時間はほぼ2倍（680〜980 s）です。
  - 粗い探索（16 / 8 µm、E2 c16f8）では100 s、72 MBでした。verdictは8 / 4 µmと同じです（6_2はLOW_PEAKでfallback、8_6はAMBIGUOUS）。
  - 4 µmの特徴量（E2 feat4）は、時間はほぼ同じでした。6_2ではmarginが0.12から0.11に下がった一方で、peakのzが6を少し超え（6.16）、UNIQUEとして採択されました。判定の閾値付近では、解像度の違いだけでverdictが変わります。
- IF Fixedでは、大きいcanvasをmovingで回すのでメモリが約1.3倍になります。
- 大きな画像に対して点群・特徴量のgridは物理単位で作るので、コストは撮像面積とgrid解像度で決まります。画素数には依存しません。

## 2. 質問への回答

**Q1. 既存Point-only Pipelineの性能を完全に維持できたか。**
はい。既存A2＋ICPは無変更で、wrapperを通しても、画像を一緒に渡しても、結果は既存実データ・新サンプルの両方で差0.0です（§1.1）。Production・既存analysisのtracked 81ファイルと既存analysis 2,422ファイルのSHA-256は、作業の前後で一致しました。ただし、既存A2はもともと「視野の大きさがほぼ同じ」前提なので、新サンプルでは失敗します。これを直したE0xは画像を使わない新しい探索経路で、E0を置き換えるものではありません。

**Q2. 画像情報は対応領域の発見に追加価値を持つか。**
限定的です。6_2では、画像だけ（核map、またはtissue＋核map）でも正しい領域をUNIQUEで見つけました。これは点群とは独立した確認になります。しかし点群（E0x）だけでも同じ領域をmargin 0.48で見つけており、画像を加えても見つかる領域は変わりません。8_6では、画像を加えても曖昧さは解けませんでした。半合成でも、画像のみ（E2）は点群より不安定です（50%）。

**Q3. 片側画像だけでも利用価値があるか。**
位置合わせの精度には効きません（E1はすべての条件でE0xと同じ解）。価値があるのは補助的な面です。

- H&E側の核を画像から検出できる（今回のStarDist）
- tissue mask・撮像範囲による品質確認
- 画像↔核の整合の監査
- IF画像を使ったcross layerでmarginが上がる（0.48 → 0.57）

一方、HE画像のcross layerではmarginが下がりました（0.27）。片側画像が常に役立つとは言えません。

**Q4. 両画像を使用することで位置合わせ精度は改善するか。**
しません。6_2で採択されたE2 / E3は、すべてICPの後でE0xと同じ解（NN median 1.91 µm、≤3 µm 0.599）に収束しました。画像を使うと探索のmarginは上がることがあり（E3 rerankで0.62）、点群と独立した確認にもなります。しかし、最終的な精度は点群とICPで決まっています。

**Q5. H&Eと蛍光のどちらをFixedにしても動作するか。**
はい。6_2では、HE FixedとIF Fixedの結果の差は、E0x・E2・E3のいずれでも中央値0.23 µm（最大0.44 µm）で、同じ対応を表しています。Affine2Dの順写像・逆写像の往復と合成のテストは `test_oir.py` で通っています。8_6でE0x / E3が入れ替えで大きく変わるのは、AMBIGUOUSで解が定まっていないためです（どちらも不採択）。なお、既存E0は入れ替えで399 µm変わり、方向に依存します。

**Q6. 画像処理に失敗した場合、安全にPoint-onlyへ戻れるか。**
はい。6_2のE2（LOW_PEAK / AMBIGUOUS）とE3 zminでは、`POINT_ONLY_FALLBACK` でE0xの結果を採用し、その事実と理由をartifactに記録しました。

- 点群のfallbackも不採択の場合（8_6）は、変換を採択せず「no transform accepted」と記録します。
- 画像scoreが高くても核配置が不整合なら採択しない、という規則は、陰性対照と半合成absentで誤採択0件でした。
- 既存のA2 / ICPの安全性検出器は緩めていません。採択の閾値も既存と同じ値です。

**Q7. 大きさが異なる画像のPartial-overlap Registrationとして、現在の設計は妥当か。**
同じ切片では妥当です。

- 撮像範囲を物理単位で扱う
- canvasを和集合サイズにする
- ICPを相手の撮像範囲内の点に限る
- AMBIGUOUS / LOW_PEAKを返す

この設計で、6_2は解け、半合成ではq 0.3まで100%正しく採択できました。別の切片（8_6）では解けていませんが、誤った候補を採択せずに止まっています。課題は次の3つです。

- 似た構造が繰り返す密な組織では、margin 0.10の規則では判断できない
- 原点（stage座標）がUNKNOWNなので、事前の位置情報を使えない
- 切片間の距離がUNKNOWN

**Q8. Workflow C/Dへ統合する場合、どのインターフェースだけを変更すればよいか。**
`architecture.md` §6のとおり、変更点は次の3つだけです。ICP本体・A2・Graph/UOT/B2・safety gateは変更しません。

1. Workflow Cの入力parserを `Sample`（frame＋核＋任意の画像）にする
2. 探索の入口を `PoseSearch` にする（点群のみ / 画像あり共通）
3. 出力を `TransformArtifact` に拡張する（frame、forward / inverse、validity region、image_status、fallback、accepted）

Workflow Dは、artifactの `inverse_transform` とframe情報を読んで、既存の逆写像resamplingを適用するだけです。Workflow Dは変換を再推定しません。点群のみの経路は `mode=P0`、`images_used_in_search=[]` として、既存と同じ挙動を保ちます。

## 3. 最終判断

- **主結論：A. POINT-ONLY REMAINS PREFERRED**
  - 6_2では、点群のみ（E0x）で正しい領域と精度が得られ、画像を足しても精度は変わりません。
  - 半合成でも、点群（E0x）が最も安定しました。
  - 画像は、点群と独立した確認・品質確認・fallbackの判断に使う補助として位置づけます。
- **副結論：**
  - **F. REGION LOCALIZATION STILL UNRESOLVED**（031-8_6）：別切片の密な組織では、点群・画像のどちらでも対応領域が決まりませんでした。
  - **E. IMAGE QUALITY / METADATA LIMITATION**：
    - 031-8にはH&Eがない（031-7で代用）
    - 原点・切片間の距離がUNKNOWN
    - 蛍光は表示用RGBで、純粋な核チャンネルがない（B−Rで代用）
- B / C / Dは採用しません。画像を追加して最終結果が良くなった場面はありません。E3が単独の手法より勝ったのは、E2の失敗を点群が補った場面だけです。

## 4. 限界

- 実データの両画像付きサンプルは2件で、そのうち正しく解けるのは1件（6_2）です。6_2の参照も独立した正解ではありません。
- 6_2の参照（提供者のNCC）とE0xの差2.2 µmのうち、どちらがより正しいかは決められません。
- 半合成の生成器は、疑似蛍光の表現を単純化しています（§1.6）。E2の失敗率、特に6_2由来の0/12は、実データの性能をそのまま表すものではありません。
- 探索の判定規則（z 6、margin 0.10）は事前に決めたもので、今回のデータに合わせていません。そのため、6_2のE2（tissue）のように、位置は正しくてもLOW_PEAKで不採択になる場合があります。
- 研究用のプレビュー（`previews/`）は確認用です。Production Workflow Dの出力ではありません。採択されなかった変換（8_6）のプレビューは、失敗例として作ったものです。

## 5. 成果物

| 種類 | ファイル |
|---|---|
| 入力監査 | `input_audit.md`、`sample_manifest.json`、`coordinate_consistency.csv` |
| 設計 | `architecture.md`、`oir/`（frames / features / search / pipeline / synthetic / metrics） |
| 結果 | `mode_comparison.csv`、`region_search_metrics.csv`、`registration_metrics.csv`、`runtime_metrics.csv`、`synthetic_results.csv`、`synthetic_summary.csv`、`diagnostic_summary.json` |
| Transform Artifact | `artifacts/<sample>_<config>.json` |
| 図 | `input_image_overview.png`、`image_nuclei_overlay.png`、`structural_feature_maps.png`、`region_search_candidates.png`、`registration_overlay_before_after.png`、`point_only_vs_image_assisted.png`、`failure_and_fallback_examples.png`、`synthetic_method_comparison.png` |
| プレビュー | `previews/preview_<sample>_<config>.png`（研究用） |
| 照合 | `freeze/`（作業前後のSHA-256、Slurmのjob ID） |

再実行の手順は `README.md` にあります。commit / pushはしていません。
