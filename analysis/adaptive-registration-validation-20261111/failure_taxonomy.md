# 成功・失敗の分類（2026-11-11）

- 対象は、16 ケース × 方式 A〜D。
- 出典は `localization_comparison.csv`、`validation_rows.csv`、`quadrant_audit_all_candidates.csv`、`profiles.csv`。
- 全ケースが開発データ（031-6_2、031-8_6）から作った派生ケースである。
- 未使用の実画像ペアはなかったため、ここで述べる「失敗原因」は、**派生条件の範囲で観測した原因**である。

## 評価の区別
- **自動指標**：
  - 探索の判定（verdict）、margin、核の近接率、象限一致
  - 参照姿勢からの probe 距離。`at_reference` は 100 µm 以内かどうかで、この閾値は事前に決めた。
- **ランドマーク評価**：8_6 由来のケースだけ。ユーザーが 10-27 に登録した 8 点を使う。LOO 中央値 23.7 µm、完全な正解ではない。
- **目視評価**：今回は実施していない。誤採択の 1 件（N6_2_IFmirror）は、構成上、鏡像なので誤りであることが確定している。
- 6_2 の参照姿勢は、自動で採択された変換である。独立な正解ではない。

## ケースごとの分類

| ケース | 期待 | A 核探索 | B tissue+nuclear | C occ+cleft | D 本番の判定 | 主な原因の分類 |
|---|---|---|---|---|---|---|
| P8_6_orig | 正 | ✘ AMBIGUOUS、1255 µm | ✘ AMBIGUOUS、1202 µm | ✔ UNIQUE、23 µm | TISSUE_LEVEL ✔ | A：**核情報の不一致**（隣接切片）。B：**不適切な画像チャンネル**（IF の tissue は核が支配的） |
| P8_6_IFhalfL | 正 | ✘ 1232 µm | ✘ 932 µm | ✔ UNIQUE、21 µm | TISSUE_LEVEL ✔ | 同上 |
| P8_6_IF500 | 正 | ✘ 1022 µm | ✘ 866 µm | △ top-1 は正しい（15 µm）が AMBIGUOUS（margin 0.029） | NOT_ACCEPTED（C1）＝偽陰性 | C：**小さい重なり領域**（FOV の比 18）で**候補が一意に決まらない**。外縁の情報がない（boundary_info 0） |
| P8_6_HEpartial60 | 正 | ✘ 1035 µm | ✘ 1202 µm | ✔ UNIQUE、26 µm | NOT_ACCEPTED（C3）＝偽陰性 | **検証での誤った棄却**：重なりが 60% で、2 象限が HE の外 |
| P6_2_orig | 正 | ✔ 2 µm | ✔ 2 µm | ✔ 2 µm | ACCEPTED ✔（同一切片のゲート） | — ただし**座標系・来歴メタデータの不備**：切片の関係が食い違っている |
| P6_2_IFhalfL | 正 | ✘ **UNIQUE なのに誤った位置**（859 µm、margin 0.881。正解は 2 位） | ✔ 4 µm | ✔ 3 µm | ACCEPTED ✔ | A：**候補ランキングの失敗**（視野を半分にすると、核の密度パターンが別の場所に強く一致した）。核の検証では止まった |
| P6_2_IF500 | 正 | ✔ 2 µm | ✔ 2 µm | ✔ 4 µm | ACCEPTED ✔ | 隣接ゲートで扱うと偽陰性（象限 2/4、**組織特徴が不足**） |
| P6_2_HEpartial60 | 正 | ✔ 2 µm | ✔ 2 µm | ✔ 5 µm | ACCEPTED ✔ | 隣接ゲートで扱うと偽陰性（**小さい重なり領域**による象限の視野外） |
| N8_6_IFmirror | 負 | AMBIGUOUS | **UNIQUE**（margin 0.146）→ 核の検証で棄却 | AMBIGUOUS | NOT_ACCEPTED ✔ | B は誤った候補が一意になった（不適切な特徴） |
| N8_6_HEexclude | 負 | AMBIGUOUS | LOW_PEAK | LOW_PEAK | NOT_ACCEPTED ✔ | — |
| N8_6_IF500xHEexclude | 負 | AMBIGUOUS | AMBIGUOUS | LOW_PEAK | NOT_ACCEPTED ✔ | — |
| **N6_2_IFmirror** | 負 | AMBIGUOUS | AMBIGUOUS | **UNIQUE**（margin 0.215、z 15.3） | 同一切片のゲート：NOT_ACCEPTED ✔ ／ **隣接ゲート：TISSUE_LEVEL ✘** | **検証での誤った採択**（隣接ゲート）。原因は**反復はしないが、大きなスケールの構造（外縁と空白）が支配的**なこと。象限 3/4 |
| N6_2_HEexclude | 負 | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS | NOT_ACCEPTED ✔ | — |
| N6_2_IF500xHEexclude | 負 | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS | NOT_ACCEPTED ✔ | — |
| Ncross_HE6_2xIF8_6 | 推定負 | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS | NOT_ACCEPTED ✔ | — |
| Ncross_HE8_6xIF6_2 | 推定負 | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS | NOT_ACCEPTED ✔ | — |

## 原因カテゴリ別の集計（この範囲で観測したもの）

| カテゴリ | 観測した例 | 備考 |
|---|---|---|
| 不適切な画像チャンネル | B が 8_6 系の 4 ケースすべてで失敗 | 10-27 の診断を再現した。occ+cleft で解消 |
| 核情報の不一致（隣接切片） | A が 8_6 系の 4 ケースすべてで失敗（約 1000〜1250 µm） | 隣接切片では核の点群探索は使えない |
| 小さい重なり領域、小視野 | P8_6_IF500（C1 AMBIGUOUS）、HEpartial60（C3 で棄却） | 偽陰性になる（安全側） |
| 候補ランキングの失敗 | P6_2_IFhalfL の A（誤った位置が UNIQUE） | 核の検証で止まった。同一切片の核ゲートが有効に働いた |
| 検証での誤った棄却 | HEpartial60 ×2、IF500 ×2（隣接ゲート） | 象限チェックが部分重なりに対応していない |
| 検証での誤った採択 | N6_2_IFmirror（隣接ゲートのみ） | TISSUE_LEVEL の危険性 |
| 座標系・来歴メタデータの不備 | 6_2 の切片の関係（DIFFERENT か SAME か） | 本番のゲート選択が、ファイル名の推測で決まっている |
| 組織特徴の不足、組織の占有率の低さ、反復構造、大きな回転・倍率の差、局所変形、画像品質の不良 | **観測できなかった**（該当するデータがない） | 回転は全周を探索している。scale は 0.95〜1.05 × 微調整 0.97〜1.03 で、それより大きい差には対応していない |

## occ+cleft について（Phase E の特記事項）
- 8_6 に似た条件（組織がほぼ全面で、間隙がはっきりしている。IF の occ と核の相関が負）では、派生した 4 つの正例すべてで top-1 が正しかった。
  - 小視野の 1 ケースは AMBIGUOUS で止まった。
- 失敗したのは小視野（500 µm 角）のケース。特徴 map とスコアの状況は次のとおり：
  - top-1 の score 11.7 と 2 位の差は 0.029 で、ほぼ横並び。
  - 誤った候補 5 個の score も 11 前後。
  - 500 µm 角には、間隙の網目が数本しか入らない。HE 全域（約 2.1 mm 角）には、同じくらい一致する配置が多数ある。
  - 外縁の情報（boundary_info）は 0 で、手がかりは間隙だけ。
- 6_2 由来の鏡像ケースで誤った候補が UNIQUE になったのは、occ+cleft の弱点である。
  - IF 視野の大きな外縁・空白が、鏡像でも HE 側の外縁に合ってしまった（θ=210°）。
  - この候補の score は 15.3 で、正しい候補（18.2〜25.9）より低かった。ただし、事後に分かったことなので、閾値には使っていない。
