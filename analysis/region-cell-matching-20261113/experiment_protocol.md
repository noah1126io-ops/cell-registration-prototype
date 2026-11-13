# 実験計画（2026-11-13、結果を見る前に固定）

この文書は、実験の結果を見る前に書いた。変更する場合は §9 の変更履歴に、理由と時点（どの結果を見た後か）を記録する。

## 1. 問いと仮説

- **H1（Phase 1）**：C1 の上位 k 候補それぞれに、既存の Graph / UOT / B2 を当てはめる。それで得た核対応の指標で候補を並べ替えると、C1 の 1 位が誤っているケースで正しい領域を 1 位に戻せる（救済できる）。
- **H2（Phase 1）**：正しい候補が分かれば、その候補での核対応の成績は、C1 の 1 位をそのまま使うより良くなる。
- **H3（Phase 2）**：信頼できる対応が十分にあるときは、それをアンカーにした局所変形で、Affine より独立指標（EPE、独立ランドマークの TRE）が改善する。
- いずれも「改善する」ことを前提にしない。否定された場合もそのまま記録する。

## 2. データ

| 名前 | 内容 | 用途 | 正解 |
|---|---|---|---|
| SYN-C1 | E0x の難易度生成器 `synth.make_case`（10-20、変更なし）。13 条件 × seed 3 個（1001 = 開発用、1002 と 1003 = 評価用） | 実験 B、実験 C | 真の姿勢、共有核の ID |
| SYN-FIELD | UOT 検証の合成データ `validation_lib.generate_synthetic_sample`（09-11、50 シナリオ、`validate_synthetic.scenario_table()`） | 実験 A | 真の変形場、独立ランドマーク、既知の対応 |
| SYN-RL | B2 の Real-like 生成器 `generate_real_like_sample` の **Test 分割のみ**（seed 4101〜4105 の 5 ケース。強度は Weak / Moderate / Strong を順に割り当て。B2 の学習と閾値決定には使われていない） | 実験 A | 真の変形場、ID |
| REAL-D | 11-11 の派生ケース 16 件（031-6_2 / 031-8_6 から作ったもの）。C1 / C2 の候補は、11-11 の `runs/*__D/result.json` から読む | 実データでの Phase 1、Phase 2 | 細胞 ID の正解はない。領域の参照は 11-11 の `cases.json`（probe ≤100 µm） |
| REAL-S | `serial_section_pair_20260904`（点群のみ。既存の A2 R2 による Affine。UNVERIFIED） | 実データでの Phase 2 | なし |

- 031-6_2 は、切片関係の来歴（別切片）とファイル名による推測（同一切片）が**矛盾したまま**なので、同一細胞の正解としては使わない。
- 031-8_6 は別切片で、核の一対一対応を仮定しない。
- 実データからは「対応領域の幾何学的な妥当性」と「細胞が同一かどうか」を分けて報告し、後者については主張しない。

## 3. Phase 1：候補の再評価（実験 B、実験 C）

### 3.1 候補
- SYN-C1：今のエンジンの点群探索（`WorkflowConfig(localization="points")`、`n_candidates=6`）と、C2（同一切片のゲート。ICP を使う）を実行する。候補の姿勢には `T_final` を使う。
- REAL-D：11-11 の D の実行から、`gates.same.candidates` を読む（image と points の両方、`T_final`）。8_6 は隣接切片なので、本番では `T_search` を使っている。そのため 8_6 では、`T_search` の結果も参考として記録する。

### 3.2 候補ごとの処理（`src/registration_research/cell_correspondence.py`）
1. Moving の核を候補の姿勢で Fixed 座標に写す。
2. 撮像範囲の重なりに入る核だけを使う（Fixed は写した Moving の視野の内側。Moving は Fixed の視野の内側）。
3. どちらかの核が 40 個未満なら `TOO_FEW_NUCLEI` として評価しない。
4. 凍結済みの B0 / B1 / B3 と、B2（10-14 に保存した学習済みモデルと閾値。再学習しない）を実行する。
5. **mirror decoy**：重なりの Moving の核を、重心を通る縦軸で鏡映する（C3 の mirror null と同じ）。これに同じ処理をかける。

### 3.3 再順位付けの規則（事前指定）
- **主な順位づけのスコア**：`S_excess = B2 の採択率 − decoy の B2 採択率`。採択率は、B2 で matched になった核の数を、重なりにある Moving の核の数で割ったもの。
  - 重なりの面積や核の密度で変わる「偶然の採択」を、decoy で差し引く。
  - UOT の目的関数の値は、核の数や重なりの大きさで変わるので、順位づけには使わない（記述統計として記録するだけ）。
- 同じ値のときは C1 の順位を優先する。`TOO_FEW_NUCLEI` の候補は最下位にする。
- **「Graph/UOT で支持された」と判定する条件**（3 つすべてを満たすとき）：
  1. 1 位の `S_excess` ≥ 0.10
  2. 1 位の採択率 ≥ 3 × max(decoy の採択率, 0.01)
  3. 1 位と 2 位の `S_excess` の差 ≥ 0.05
- 比較のために、次の順位づけも計算する（記述のみ）：
  - C1 の順位（既存）
  - C3 の核の近接（`le_3um`。既存のエンジンが記録している値）
  - B0 の graph support の中央値
  - UOT の輸送量の比
- 開発用の seed（1001）でだけ規則を確認してよい。変更した場合は §9 に記録してから、評価用の seed を集計する。

### 3.4 評価指標
- 正しい領域：SYN-C1 は、`T_final` の grid 誤差 ≤ 20 µm（E0x の定義）。REAL-D は、probe の距離 ≤ 100 µm（11-11 の定義）。
- Top-1 の正解率、Top-k の正解率。
- **救済**：C1 の 1 位が誤りで、再順位付けの 1 位が正しい。
- **悪化**：C1 の 1 位が正しく、再順位付けの 1 位が誤り。
- **誤った支持**：負例（absent、mirror、HEexclude、cross）や誤った候補を「支持された」と判定すること。
- 核対応（SYN-C1 のみ）：matched の precision / recall / F1、unmatched の precision / recall。比較は、C1 の 1 位と、再順位付けの 1 位と、正解の候補の 3 つ。

## 4. Phase 2：局所変形（実験 A と、実験 B の延長）

### 4.1 比較する方式
アンカーの作り方（対応推定）と、変形場の作り方（変形推定）の寄与を分けて比べられるように、次の方式を置く。

| 方式 | 対応（アンカー） | 変形場 |
|---|---|---|
| A | なし | Global Affine のみ（変位 0） |
| B | 相互最近傍（≤10 µm、既存の `matched_nuclei_rbf_fine_warp` の内部） | 既存のアプリの TPS（既定値のまま） |
| B′ | 相互最近傍（≤10 µm、`mutual_nearest_pairs`） | UOT conservative と同じ linear RBF（smoothing 10、近傍 40） |
| C | B2 の matched の組 | 同上（linear RBF） |
| D | UOT conservative の重心アンカー（row mass ≥ 0.25、信頼度で縮小） | 既存の `validation_lib.run_uot` を変更せずに使う |
| E | B2 の matched の組。変位に B2 の confidence を掛ける（D と同じ縮小の考え方） | linear RBF |

- B′ と C を比べると、対応推定の違いの効果が分かる（変形場は同じ）。
- B′ と D を比べると、硬い対応（相互最近傍）と柔らかい対応（UOT）の違いが分かる。
- **安全策**（すべての方式に共通）：
  - 既存の `common_audit` を適用する（SAFETY の閾値は変えない）。通らなければ Affine に戻す（`applied=False`）。
  - アンカーが 20 個未満なら Affine に戻す（`ANCHORS_INSUFFICIENT`）。
- Global の姿勢は固定する（実験 A では既知の姿勢、実験 B では C1 の 1 位の `T_final`）。

### 4.2 評価指標
- EPE：評価用マスクの内側での変形場の誤差（中央値、p95）。
- 独立ランドマークの TRE（SYN-FIELD）。
- 既知の対応点の TRE。実験 B では、アンカーに使わなかった共有核の TRE を主な指標にする。
- unmatched の核の移動量。
- Jacobian（最小、p05、p95）、foldover の割合、最大変位。
- 適用率、アンカー数、アンカーの適合率（正解の対応の割合）、アンカーのカバー率。
- 実行時間。
- 判定：方式 X が Affine や D に勝ったかは、ケースごとの勝ち・負け・引き分け（EPE の差が ±0.1 µm 以内なら引き分け）と、差の中央値で判断する。安全性の指標（foldover、ゲートの不通過）が悪化したら、改善とは呼ばない。
- 実データ（REAL-S、REAL-D）：最近傍距離は細胞同定の精度として扱わない。報告するのは適用されたか、変形場の統計、そして 8_6 の独立ランドマーク（ユーザーの 8 点。マッチングと変形には使っていない）の残差の変化だけ。

## 5. 正解データの隔離
- 正解（真の ID、姿勢、変形場）は、評価の関数だけに渡す。
- 探索、マッチング、採択、再順位付け、変形のどれも正解を受け取らない。`run_pipeline` に正解の ID を渡すと、評価用の列（`top1_correct`）だけが埋まる。採択に使う特徴量には正解が入らない。これは tests_rcm で確かめる（正解を渡したときと渡さないときで、状態がまったく同じになること）。
- SYN-RL では、B2 の学習（Train）と閾値の決定（Validation）に使われた seed を使わない。

## 6. 既存の基準の維持（回帰）
- `src/registration_core`、`src/registration_explorer` の既存の動作、本番の Workflow C / D、既存の研究実装は変更しない。新しいコードは `src/registration_research/` と、この研究ディレクトリにだけ置く。
- 確認すること：
  1. 既存の `pytest tests` の結果が変わらないこと。
  2. 凍結済みの研究テスト（10-14、10-08）が通ること。
  3. 実データの B0 / B1 / B3 / B2（`serial_section_pair`）を再実行すると、10-14 の `real_exploratory_per_nucleus.csv` と一致すること。
  4. SYN-FIELD の D が、09-11 の `synthetic_metrics.csv`（UOT conservative）と一致すること。
  5. 再利用したファイルの sha256 が `baseline_manifest.json` と変わらないこと。

## 7. 実行環境
- 計算は Slurm（mib-dbia）で行う。SYN-C1 の C1 は、1 ケースあたり 4 CPU で約 7 分。Phase 1 は、1 候補あたり約 60 秒（decoy を含む）。

## 8. 成果物
`REPORT.md`、`baseline_manifest.json`、`reuse_audit.md`、この文書、`candidate_comparison.csv`、`correspondence_results.csv`、`deformation_comparison.csv`、`failure_analysis.md`、`test_results.md`

## 9. 変更履歴
- 2026-11-13（結果を見る前）：SYN-RL を「15 ケース」と書いていたのは誤りで、Test 分割は 5 ケースだった（`real_like_scenarios()` で確認）。表の記述を直した。
- 2026-11-13（**実験 C を追加**）：
  - 時点：SYN-C1 の C1 の結果（20 / 39 件。候補ごとの正しい領域のラベル）を見た後。Phase 1 の再順位付けの結果は、どのケースもまだ見ていない。
  - 見たこと：C1 の 1 位が誤る条件（overlap=0.3、repeat=0.8 と 1、ratio=0.25 ＋ ns=0.85）では、正しい領域が上位 6 件に入っていなかった。11-11 の実データの 8_6 系と同じ傾向。
  - 追加する実験（**実験 C-K30**）：エンジンの設定の上書き `WorkflowConfig.search={"n_fine": 30}` だけで、細かい段階に進める候補を 30 件に広げる。コードは変えず、本番の既定値（6）も変えない。
  - 対象：次の 7 条件 × seed 3 個 = 21 ケース。repeat=0.5、repeat=0.8、repeat=1、overlap=0.5、overlap=0.3、grid ratio=0.25 ns=0.85、absent
  - 追加で記録すること：粗い段階で正解が何位にあるか（候補の再現率）。
  - 再順位付けの規則（§3.3）と閾値は変えない。
- 2026-11-13（**「2 位との差」の条件の実装を修正**）：
  - 時点：REAL-D の P6_2_orig と N8_6_IF500xHEexclude の Phase 1 の結果を見た後。
  - 問題：image と points の候補をまとめて並べると、同じ領域が 2 回入ることがある（P6_2_orig では image#0 と points#0 が同じ領域で、採択率も同じ 0.584）。§3.3 の「1 位と 2 位の差」は、別の領域どうしを比べる意図だったが、実装では重複している候補と比べていた。
  - 修正：2 位として、1 位と**別の領域**の候補のうち最良のものを使う。同じ領域かどうかは、エンジンの `distinct` と同じ probe（Moving の中心 ±200 µm）で、距離の中央値が 40 µm 未満かどうかで判定する。閾値（0.10、3 倍、0.05）は変えない。
  - 影響：判定はすべて、保存した候補の姿勢から `collect.py` で計算し直す。並べ替えの順位そのものは変わらない。
