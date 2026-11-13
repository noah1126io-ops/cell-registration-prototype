# テスト結果（2026-11-13）

すべて Slurm（mib-dbia、GPU 1 枚、`pytest_baseline.sbatch`）で実行した。ログは `logs/pytest_<job>.out` にある（ログは git に含めない）。

| 対象 | 作業前（job 910609、HEAD dd68fca、新しいファイルのみ） | 作業後（job 911335、HEAD dd68fca ＋ 今回の変更） |
|---|---|---|
| 既存のテスト `tests/` | 270 passed, 1 skipped | **270 passed, 1 skipped**（変化なし） |
| 凍結済み 10-14 `graph-matching-adoption-separation-20261014/tests` | 10 passed | **10 passed** |
| 凍結済み 10-08 `contextual-graph-uot-20261008/tests` | 7 passed | **7 passed** |
| 今回の `tests_rcm` | （まだ作成前） | **13 passed** |

- skip の 1 件は、作業前から GPU のノードに依存しているテストで、今回の変更とは関係しない。
- 作業後の時点での既存コードの変更は、`src/registration_explorer/page.py`（研究タブの差し込み 7 行）と `README.md`（環境変数の表に 1 行）だけ。

## tests_rcm の内容（13 件）
| テスト | 確かめること |
|---|---|
| `test_reused_research_files_unchanged` | 再利用した研究ファイル 13 個の sha256 が `baseline_manifest.json` と一致する |
| `test_production_and_engine_files_unchanged` | 本番とエンジンのファイル 35 個の sha256 が一致する（変えたのは `page.py` だけ） |
| `test_frozen_b2_models_are_the_saved_ones` | B2 のモデルと閾値（match 0.8087、unmatched 0.4332）が 10-14 の保存値と同じ |
| `test_real_b0_b1_b3_b2_reproduce_2026_10_14`（slow） | 実データ（serial_section_pair）の B0 / B1 / B3 / B2 を再実行すると、10-14 の `real_exploratory_per_nucleus.csv` と**完全に一致**する |
| `test_uot_conservative_reproduces_2026_09_11` | SYN-FIELD の D（UOT conservative）が、09-11 の `synthetic_metrics.csv` と一致する |
| `test_candidate_adapter_reads_engine_records_unchanged` | 候補のアダプタが、エンジンの記録（姿勢、判定、C3）を変えずに読む |
| `test_truth_never_changes_matching_or_adoption` | 正解の ID を渡しても渡さなくても、B0 / B1 / B3 / B2 の状態と対応先が完全に同じ（正解の漏れがない） |
| `test_mirror_decoy_keeps_density_and_destroys_identity` | decoy は核の数と重心を保ち、元の配置とは異なる |
| `test_too_few_anchors_falls_back_to_affine` | アンカーが 20 個未満なら Affine に戻る（`ANCHORS_INSUFFICIENT`） |
| `test_reranking_rule_is_the_prespecified_one` | 再順位付けの閾値が事前指定の値（0.10、3 倍、0.05）のまま。主指標が 0.10 未満なら支持しない。`TOO_FEW_NUCLEI` は最下位（同じ領域の重複を除く処理は、このテストでは確かめていない） |
| `test_explorer_research_tab_is_off_by_default` | 環境変数がなければ研究タブは無効 |
| `test_explorer_page_keeps_existing_tabs_when_disabled` | 無効のとき、既存の 8 つのタブがそのまま残る |
| `test_research_tab_renders_saved_results_read_only` | 有効にすると、031-6_2 と 031-8_6 の保存済みの結果を Streamlit AppTest で表示でき、例外が出ず、`runs/p1_*.json` の sha256 が変わらない |

## 数値の再現（回帰の確認）
- SYN-FIELD 50 シナリオの D の EPE の中央値は 2.289 µm で、09-11 の値（2.2889 µm）と一致した。
- REAL-S の D の最近傍距離の中央値（10.02→8.63 µm）は、09-11 の実データの値と一致した。
- 既存の研究実装（10-08、10-14、09-11）と本番のファイルは、どれも変更していない（sha256 で確認）。
