# 研究の進捗と引き継ぎ（エージェント向け）

- 最終更新：2026-11-13（region-cell-matching-20261113 の作業が終わった時点の状態）
- 対象：このリポジトリで作業を引き継ぐエージェント（Claude Code などの AI エージェント）
- 位置づけ：研究の現状、決まったこと、未解決の問題、作業のルールをまとめた入口です。詳しい内容は、各研究ディレクトリの `REPORT.md` にあります。
- 更新のしかた：研究のタスクが終わるたびに、この文書の §2〜§6 を更新してください。過去の記述は消さずに、§7 の履歴に追記してください。

---

## 0. まず読むもの（この順）

1. この文書
2. `README.md` の「Registration Explorer (Research) と研究Workflow C」
3. 最新の研究：`analysis/region-cell-matching-20261113/REPORT.md`（核対応による候補の再評価と局所変形。結論は §3）。ロードマップは `analysis/adaptive-registration-validation-20261111/REPORT.md` と `v1_roadmap_recommendations.md`
4. サンプルの来歴：`analysis/e0x-difficulty-and-web-integration-20261020/sample_provenance.md`
5. エンジンの本体：`src/registration_core/workflow.py`（`CORE_VERSION`、`WorkflowConfig`、`infer_section_relation`、C3 の規則）

## 1. 研究の目的

- 異なる切片の HE 画像と蛍光（seqIF）画像について、次の 3 つを行う。
  - 対応する組織領域を自動で探す。
  - 核の位置で位置合わせ（global affine）を精密にする。
  - 事前に決めた規則で、その結果を「採択してよいか」判定する。
- 採択された結果だけを、Production の Workflow C / D（細胞レベルの解析）に渡す。
- 最も大事な要件は、**誤った位置合わせを採択しないこと（誤採択 0）**。見逃し（採択できない）は、手動の補助で補う方針。
- 目標：研究用 v1.0 を 2026-12-31 に完成させる（`v1_roadmap_recommendations.md`）。

## 2. 現在の状態（2026-11-13）

| 項目 | 状態 |
|---|---|
| ブランチ | `feature/gpu-slurm`（worktree `projects/cell-registration-prototype-gpu`）。HEAD は git log で確認する（11-13 の作業はコミット・push 済み） |
| エンジン | `registration_core 1.2 (2026-11-10)` |
| 実行中のジョブ | なし |
| 未完了の作業 | なし。次の作業はユーザーの承認待ち（§5） |
| テスト | `pytest tests`：270 成功、1 スキップ。条件は §6 を参照。研究テスト：`analysis/region-cell-matching-20261113/tests_rcm`（13 件） |
| 研究用コード | `src/registration_research/`（C1 の候補アダプタ、凍結済み Graph/UOT/B2 の呼び出し、再順位付け、局所変形の比較）。本番からは使われない |
| Explorer の研究タブ | 「核対応（研究）」。既定で無効（`CELLREG_REGION_CELL_MATCHING=1` で有効）。保存済みの結果を表示するだけ |

### エンジン（`src/registration_core`）の処理の流れ
1. **C1 対応領域の探索**：画像の探索と、核点群の探索の両方を行う。
   - 画像の特徴は `occ+cleft`（1.1 から既定）。IF の非核チャンネル max(G,R) から作った占有率と、組織の隙間（cleft）の帯域通過。
2. **C2 精密化**：重なり領域の核だけで ICP を行う。
3. **C3 検証**：切片の関係によって、使う規則が変わる。
   - 同一切片（same）：UNIQUE、かつ ≤3 µm ≥ 0.25、かつ偶然一致の 3 倍以上 → `ACCEPTED`
   - 隣接切片（adjacent）：核のチェックは記録だけ。IF を 2×2 の象限に分けて別々に探し直し、3/4 以上が 30 µm・2° 以内なら → `TISSUE_LEVEL`（`accepted=False`、Workflow D には渡さない）
   - 切片の関係は、現在は**ファイル名から推測している**（`infer_section_relation`）。これが §4 の問題の原因。
4. **C4 記録**：`artifact.json`（`cellreg-registration-artifact/1`）に保存する。

### サンプル（画像のある HE+IF の組は、この 2 組だけ）

| サンプル | 切片の関係 | 現在の判定 | 備考 |
|---|---|---|---|
| 031-6_2 | manifest では**別切片**（10-20 のユーザー訂正）。エンジンはファイル名から**同一切片**と推測している → **CONFLICT** | ACCEPTED（1.9078 µm） | 判定がファイル名に左右されている（§4） |
| 031-8_6 | 隣接切片（HE は 031-7、IF は 031-8。切片間の距離は不明） | TISSUE_LEVEL（4/4 象限が ≤10 µm） | ユーザーのランドマーク 8 点の fit と、自動結果のずれは約 24 µm。ランドマーク自体の精度（LOO の中央値 23.7 µm）と同程度 |

## 3. これまでにわかったこと（要点）

- **8_6 が以前失敗していた原因**は、IF の組織 map が核（Hoechst）に支配されていたこと。HE と IF は隣接切片なので、核の位置は対応しない。→ `occ+cleft` で解決（10-27）。
- **核の ≤3 µm 規則は、隣接切片には使えない**。8_6 では、正しい姿勢での値（0.074）と、誤った姿勢での上限（0.076）が区別できない。→ 隣接切片用の象限チェック（TISSUE_LEVEL）を追加（11-10）。
- **未使用の実データは 0 件**（11-11 に調査）。独立した外部検証はまだできていない。2 組から作った派生ケース 16 件の結果は次のとおり。これは独立した検証ではない。
  - occ+cleft は、正例 8 件すべてで正しい領域を 1 位に選んだ。核探索（A）と旧特徴（B）は、8_6 系では約 870〜1250 µm ずれた。
  - 本番の判定（D）：誤採択 0 件、見逃し 2 件。
    - IF500：候補を 1 つに絞れなかった（AMBIGUOUS）。
    - HEpartial60：象限チェックが 2/4（HE の外にはみ出した象限が失敗と数えられる）。
- **過学習の注意**：occ/cleft は 8_6 を見ながら設計し、正規化は 6_2 を見た後に決めた。独立と言えるのは合成データのテストだけ。
- 合成データでの難易度の掃引（10-20、132 ケース）：誤採択 0 / 177 実行。反復構造の割合が 0.5 以上、重なり 0.3、共有されない構造 0.85、変形 10 µm では失敗する。
- **核対応（Graph/UOT/B2）で C1 の候補を並べ替えても、救済はほぼできない**（11-13）。
  - 合成 39 件で救済 0 件。実データ由来では、points の候補だけの場合に 1 件で、それは既存の C3 でも救えた。
  - 主な原因は、正しい領域が上位 6 件に入っていないこと（候補生成の再現率）。
  - 候補の識別力は既存の C3 と同程度（AUC 0.972 と 0.977）。周期的な組織では、誤った領域でも B2 が 6 割を matched にする。
- **局所変形**（11-13）：Global が正しい合成データでは、B2 の対応をアンカーにした linear RBF（C）が最良（EPE 2.92→0.89 µm）。
  - ただし、変形が小さいときは Affine を悪化させる。
  - 8_6 のランドマークの残差は改善しない。
  - 既存の監査は領域の正しさを見ないので、誤った領域でも B′ / D の局所変形が適用される。

## 4. 未解決のリスクと問題

1. **6_2 の切片関係の CONFLICT**：manifest の `provenance.same_section` は `false`。一方、エンジンはファイル名（`031-6_2_HE_031-6_XY02`）から same と推測している。
   - 別切片が正しいなら、6_2 の正しい判定は TISSUE_LEVEL で、細胞レベルの採択はしてはいけない。
   - 11-10 の `REPORT.md` で 6_2 を「同一切片」と書いたのは誤り。過去の analysis は上書きしないルールなので、訂正は 11-11 の `REPORT.md` と `sample_provenance.md` にある。
2. **鏡像の誤採択の危険**：6_2 の IF を鏡像にした負例が、隣接切片の規則では画像 UNIQUE ＋ 象限 3/4 → TISSUE_LEVEL まで通ってしまう。本番は 6_2 に同一切片の規則を使うので、今は止まっている。切片の関係を直すと、この問題が表に出る。
3. **Explorer のストア**に残っている 8_6 の結果は、10-27 の古い実行のまま。UI からの再実行はユーザーに任せている（ストアへの書き込みになるため）。
4. **artifact が `git_dirty=True` で記録されている過去の実行**がある。11-11 にすべてコミットしたので、今後の実行は clean な状態で行える。
5. **局所変形を本番に入れる場合**は、領域の検証を前提条件にする必要がある。誤った領域でも、既存の `common_audit` は B′ / D の変形を通す（11-13、failure_analysis.md §3.1）。
6. **UOT conservative が 6_2 で数値的に失敗する**（translation-invariant な Sinkhorn がほぼ 0 の輸送量を返す）。研究ラッパーは Affine に戻している。既存の `run_uot` は直していない。

## 5. 次の作業（ユーザーの判断待ち）

最優先は **A1 と A8**（`v1_roadmap_recommendations.md`）。どちらも**着手前にユーザーの承認が必要**。本番の既定動作が変わり、6_2 が ACCEPTED から保留になるため。

- **A2（人の作業、A1 の前提）**：031-6 XY02 の HE と seqIF 031-6_2 が同一切片かどうかを、ユーザーまたはデータ提供者に確認する。
- **A1**：`WorkflowConfig.section_relation` を manifest の `provenance.same_section` から設定する。
  - ファイル名からの推測は「ヒント」として記録するだけにする。
  - 来歴と食い違ったら `CONFLICT` とする。
  - `UNKNOWN` と `CONFLICT` のときは、細胞レベルの自動採択を保留する。
- **A8**：回帰テストを追加する。
  - 6_2 / 8_6 の変換が 11-10 と同一であること。
  - 鏡像の負例が本番の規則で NOT_ACCEPTED になること。
- その後の作業：A3（Data Profiler を artifact に入れる）→ A4（戦略の提案）、A5（判定理由の表示）→ A6（失敗時の手動補助）、A7（再現できる実行履歴）、A9（IF チャンネルの設定化）。
- 11-13 の研究からの候補（どれも承認待ち、優先度は低い）：
  - C1 の候補生成の再現率を上げる（K30 の結果は 11-13 の REPORT §4）。
  - B2 アンカーの局所変形を、領域の検証後、同一切片に限って条件付きで使う研究。
- 外部検証には新しいデータが必要（仕様は `v1_roadmap_recommendations.md` の「必要なデータ」）。データが届いたら、`data_profiler_spec.md` の §5 の手順と、`benchmark_runner.py` / `collect.py` で再実行できる。

## 6. 作業のルール（必ず守る）

### ユーザーとのやりとり
- 進捗報告やユーザーへの返信は**日本語**で書く。ファイル名、コマンド、指標名は英語のままでよい。
- 本番の既定動作を変える変更、ストア（`research.sqlite`）への書き込み、データの削除は、事前にユーザーの承認を得る。

### Git
- **タスクが終わるたびに commit と push** を行う（`origin feature/gpu-slurm`）。作業の途中では push しない。`main` には、ユーザーに確認せずに push しない。
- 含めるもの：コード、テスト、レポート、5 MB 未満の小さな成果物（.py/.md/.csv/.json/.png/.svg/.sbatch）。
- 除外するもの（`.gitignore` に設定済み）：`analysis/**/cache/`、`*.npz`、`*.npy`、`*.tif*`、`*.sqlite`、5 MB 以上のファイル、`runs/`。
- `gh` はインストールされていない。ssh での push はできる。
- git の stash はほかの worktree と共有されている。`git stash` / `git stash pop` をそのまま使わない。

### 研究の進め方
- 新しい研究は `analysis/<テーマ>-YYYYMMDD/` に作る。`REPORT.md` を書き、判定の規則は**実行前に固定する**（例：`PRESPECIFIED_RULE.md`、`protocol.md`、`baseline_manifest.json`）。
- **過去の analysis ディレクトリは上書きしない**。訂正は新しいレポートか `sample_provenance.md` に書く。
- 結果を「正しい」と判定する基準（例：probe 距離 ≤100 µm）は、実行前に決めてコードに書く。

### 計算環境
- 重い計算は Slurm で実行する（パーティション `mib-dbia`）。ゲートウェイで動かすのは軽い確認だけ。
- 計算ノードからは `/tmp` が見えない。結果は srun の標準出力か、共有ディスク上に保存する。
- Python：`/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python`
- エンジンの実行は遅い：1 ジョブあたり 25〜54 分（4 CPU）。Slurm の配列ジョブは `%8` で同時実行数を制限する。
- テスト：`pytest tests` で実行する。リポジトリのルートでは実行しない（`analysis/` にテストのような名前のスクリプトがある）。
  - GPU parity テスト 2 件は、`--gres=gpu:1` がないと失敗する（`CUDA_VISIBLE_DEVICES` が未設定になるため。コードの問題ではない）。
  - 例：`analysis/adaptive-registration-validation-20261111/pytest.sbatch`

### Explorer（開発サーバー）
- `analysis/e0x-difficulty-and-web-integration-20261020/dev_server.sh start|stop|status 8511`
- コードを変えたら `stop` → `start` で反映する（file watcher は無効）。
- ユーザーの PC からは、ゲートウェイのホスト名を解決できない。IP を指定して SSH でポート転送している（`ssh -N -L 8511:127.0.0.1:8511 <user>@<gateway の IP>`。IP はユーザーに確認する）。「サーバーが止まった」と言われたら、まず `status` を確認する。多くの場合、原因はポート転送が切れたこと。
- E2E テスト用の Playwright 環境は、古いセッションの scratchpad にあった。消えている可能性がある。

## 7. 研究の履歴

| 日付 | ディレクトリ（`analysis/`） | 要点 |
|---|---|---|
| 09-08 | （`docs/WORKFLOW_C_SWEEP*.md`） | Workflow C の GPU パラメータ掃引、Slurm worker |
| 09-11〜10-15 | `uot-adoption-validation-20260911`、`contextual-*-20261008`、`affine-*-2026101x`、`graph-matching-*`、`real-*-2026101x` | 点群による位置合わせ（UOT、グラフマッチング、A2 affine）の検証 |
| 10-19 | `optional-image-assisted-registration-20261019`、`he-stardist-seqif-registration-20261019` | 点群のみ（E0x、サイズを考慮した探索）が優先。8_6 は未解決（HE が 031-7） |
| 10-20〜22 | `e0x-difficulty-and-web-integration-20261020` | 合成データの難易度掃引（132 ケース、誤採択 0）。Registration Explorer の UI と手動調整。**6_2 は別切片（ユーザー訂正）** |
| 10-22 | `workflow-c-unification-and-registration-review-20261022` | 共通エンジン `src/registration_core`、SQLite の研究ストア、Explorer の研究環境。6_2 ACCEPTED、8_6 は C1 で NOT_ACCEPTED |
| 10-27 | `real-image-localization-failure-audit-20261027` | 8_6 の失敗原因は、核に支配された IF の特徴。`occ+cleft` を既定にした（core 1.1）。参照変換の UI。ユーザーがランドマークを 8 点登録 |
| 11-10 | `landmark-reference-failure-analysis-20261110` | 核の C3 規則は隣接切片に使えない。象限チェックで 8_6 → TISSUE_LEVEL（core 1.2） |
| 11-11 | `adaptive-registration-validation-20261111` | 未使用データ 0 件。派生ケース 16 件でのストレステスト。鏡像のリスクと、6_2 の CONFLICT を発見。適応型レジストレーションの設計と v1.0 のロードマップ。初めて GitHub に push（`e79a8ac`..`82f1376`） |
| 11-13 | — | この引き継ぎ文書を作成 |
| 11-13 | `region-cell-matching-20261113` | C1 の上位 k 候補を Graph/UOT/B2 で再評価：救済はほぼない（候補生成がボトルネック）。B2 アンカーの局所変形は、合成データでは最良だが、変形が小さいときや誤った領域では危険。研究用コードは `src/registration_research/`、Explorer の研究タブは既定で無効 |
