# 構造化フィードバックの形式（Review Annotation と注釈）

人による評価は **Review Annotation** として保存し、正解（ground truth）としては扱いません。採択の判定も変えません。

## 1. 候補の評価（`reviews`）

| 項目 | 値 |
|---|---|
| judgement | `correct`（正しいと思う）/ `incorrect`（間違っていると思う）/ `unsure`（判断できない） |
| problems（複数可） | `region`（対応領域が違う）、`rotation`（Rotationが違う）、`scale`（Scaleが違う）、`translation`（Translationが違う）、`nuclei`（核配置が合わない）、`image`（画像構造が合わない）、`display`（表示・座標系に問題がある）、`other`（その他） |
| comment | 自由記述（任意） |
| 紐づけ | sample_id、session_id、run_id、candidate_id、transform_id |
| view（評価したときの表示） | result_id、candidate、stage（ICP後 / 探索pose）、fixed / moving、manual（手動調整中か）、表示中の変換 M・t、label |

定義にない値はコードで拒否します（`ValueError`）。

```json
{"judgement": "incorrect", "problems": ["region", "rotation"], "comment": "腺管の配置が合わない",
 "run_id": "20261022-155543-dfafb7", "candidate_id": "image#0",
 "view": {"fixed": "HE", "moving": "IF", "stage": "ICP後", "manual": false, "M": [[...],[...]], "t": [..., ...]}}
```

## 2. 注釈（`annotations`）：次の実験で再利用できる座標データ

| kind | geometry / payload | 作り方（UI） | 次の実行での使われ方（明示的に選んだ場合のみ） |
|---|---|---|---|
| `search_roi` | polygon（Fixed frame）。payload.frame_key | Viewerで「領域（ROI）を指定」→ドラッグ→送信 →『探索範囲の制限』で保存 | Fixed側の核と構造mapをROI内に限って探索（`Constraints.search_roi`）。検証は全データで行う |
| `roi` | polygon（Fixed frame） | 同上（『対応がありそうな領域』） | 探索範囲の制限としても使える（同じ変換規則） |
| `landmarks` | `[{fixed:[x,y], moving:[x,y]}]`。payload.fixed / moving | Viewerで「対応点を指定」→ Fixed側をクリック → 重ねて表示中のMoving側をクリック（Moving座標は表示中の変換の逆写像で求める）→送信 | 相似変換（Umeyama）を初期位置にして周辺を探索。すべての候補で対応点の残差を検証項目に追加 |
| `exclusion` | payload.M・t・fixed・moving（表示中の変換） | 『表示中の変換を注釈として保存』→『誤りの候補として除外』 | この配置から40 µm以内（distinct_um）の候補を除外し、残りに同じverdict規則をかけ直す |
| `manual_init` | payload.M・t・fixed・moving | 同上→『手動の初期位置』 | ±10°（1°刻み）、scale ±5%、並進150 µmの範囲で局所探索（verdict = LOCAL） |
| `note` | 自由 | — | 探索には使わない |

- 方向の扱い：Fixed / Moving を逆にして実行する場合、`exclusion` と `manual_init` の変換は逆行列にし、`landmarks` は左右を入れ替えます。Fixed frameで描いたROIを逆方向の実行に使うことは拒否します。別のframeの座標を推測で変換しないためです。
- 分離：注釈は保存しただけでは自動実行に反映されません。『新規実行』で「注釈を使った探索」を選び、注釈を明示的に選んだときだけ `Constraints` になります。その実行は `mode=assisted` として、`annotation_ids` とともに記録します。完全自動の性能評価には含めず、自動採択もしません。
- 手動変換（`transforms.kind=manual`）と対応点変換（`kind=landmark`）は、元の自動推定（`kind=automatic`）とは別の行です。親（基準の結果・候補）をdetailに記録し、元の行は変更しません。
