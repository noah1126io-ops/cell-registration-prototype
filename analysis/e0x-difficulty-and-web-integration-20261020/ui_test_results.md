# UIテスト結果（2026-10-20、2026-10-22更新）

- **ブラウザE2E**（headless Chromium + Playwright、`ui_e2e.py`、開発サーバー http://127.0.0.1:8511）：**57 / 57 通過**
- **Explorerの単体テスト**（`tests/test_registration_explorer.py`）：12 passed（2026-10-22、手動調整の2件を追加）
- **既存テスト一式**（`pytest tests`、CPUノード）：248 passed、1 skipped（2026-10-22、--gres=gpu:1 のノード）
- **GPU parityテスト**（GPUを割り当てて再実行）：3 passed（2026-10-22、--gres=gpu:1）

スクリーンショットは `ui_screenshots/` にあります。判定は、画面に表示された文字とviewerの状態だけで行っています。

## Test A：031-6_2（E0x、UNIQUE、画像・点群、Before/After、Zoom/Pan、Warp Preview）

| 確認項目 | 結果 |
|---|---|
| UNIQUE / Accepted shown | PASS |
| H&E and fluorescence images available | PASS |
| saved NN median 1.91 µm displayed | PASS |
| viewer banner accepted | PASS |
| image + point layers present | PASS |
| zoom/pan updates the viewer (HUD coordinates) | PASS |
| Before view labelled, same fixed frame | PASS |
| warp preview accepted label + inverse mapping record | PASS |
| warp uses the stored moving->fixed transform (not inverted) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/A1_031-6_2_E0x_after_overlay.png`、`ui_screenshots/A2_031-6_2_E0x_zoom_checker.png`、`ui_screenshots/A3_031-6_2_E0x_vectors.png`、`ui_screenshots/A4_031-6_2_before.png`、`ui_screenshots/A5_031-6_2_warp_preview.png`

## Test B：031-8_6（AMBIGUOUS、上位候補、切片サイズの違い、候補の切り替え、PREVIEW ONLY）

| 確認項目 | 結果 |
|---|---|
| AMBIGUOUS shown | PASS |
| PREVIEW ONLY + not passed to Workflow D | PASS |
| no Workflow D hand-off control | PASS |
| FOV of both sides + candidate regions drawn | PASS |
| candidate switch -> preview-only banner | PASS |
| candidate table with IDs and PREVIEW ONLY | PASS |
| ablation X3 candidates with ICP metrics listed | PASS |
| ablation candidate preview banner | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/B1_031-8_6_E0x_top_candidate_fov.png`、`ui_screenshots/B2_031-8_6_candidate1_preview.png`、`ui_screenshots/B3_031-8_6_candidate_table.png`、`ui_screenshots/B4_031-8_6_X3_candidates_with_icp.png`、`ui_screenshots/B5_031-8_6_X3_candidate2_icp_preview.png`

## Test C：Point-only（画像なしのサンプル、画像機能の無効化、単位の扱い）

| 確認項目 | 結果 |
|---|---|
| point-only sample: images absent, nuclei present | PASS |
| image methods disabled with reason | PASS |
| points shown without images | PASS |
| H&E frame shown in px (spacing UNKNOWN), unit mismatch noted | PASS |
| transform shown | PASS |
| 031-6_2 point-only view: E0x still available | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/C1_point_only_pair_A2.png`、`ui_screenshots/C2_031-6_2_images_hidden.png`

## Test D：Fallback（画像読み込み失敗の再現、保存済みのPOINT_ONLY_FALLBACK）

| 確認項目 | 結果 |
|---|---|
| IMAGE_UNUSABLE + point-only result kept | PASS |
| E2 saved POINT_ONLY_FALLBACK with reason | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/D1_simulated_image_failure_alert.png`、`ui_screenshots/D2_E2_point_only_fallback.png`

## Test E：Fixed/Movingの交換（両方向、逆Warp、座標の整合）

| 確認項目 | 結果 |
|---|---|
| IF-fixed result selected by default | PASS |
| viewer frame is IF | PASS |
| swap consistency table (E0x 0.23 µm) | PASS |
| inverse warp HE->IF recorded | PASS |
| IF-fixed warp uses stored HE->IF map of E0x_IFfix (not inverted) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/E1_031-6_2_IF_fixed.png`、`ui_screenshots/E2_swap_consistency.png`、`ui_screenshots/E3_warp_HE_into_IF.png`

## Test F：既存アプリとの互換性（Workflow A〜Dのページ）

| 確認項目 | 結果 |
|---|---|
| Workflow A page renders without exception | PASS |
| Workflow B page renders without exception | PASS |
| Workflow C page renders without exception | PASS |
| Workflow D page renders without exception | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/F1_workflow_c_unchanged.png`

## Test G：新規実行（Slurm投入、重複防止、キャンセル）

| 確認項目 | 結果 |
|---|---|
| A2 job not resubmitted (cached or already queued) | PASS |
| second job submitted (X0, IF fixed) | PASS |
| cancel own job 721136 (owner/name verified) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/G1_new_run_submitted.png`、`ui_screenshots/G2_run_status_after_cancel.png`

## Test H：UIから実行した結果の閲覧

| 確認項目 | 結果 |
|---|---|
| completed explorer run listed with A2 detector verdict | PASS |
| IF-frame ≤3 µm of the new A2 run equals the saved Legacy A2 value (0.0532) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/H1_explorer_run_A2_result.png`

## Test I：手動調整（回転・拡大縮小・平行移動、表示の維持、Warp Preview、保存と再表示）（2026-10-22追加）

| 確認項目 | 結果 |
|---|---|
| manual mode shows MANUAL / unverified / not passed to Workflow D | PASS |
| viewer banner MANUAL | PASS |
| identity adjustment keeps base transform | PASS |
| rotation +2° and scale x1.02 applied to the displayed transform | PASS |
| zoom kept across reruns while adjusting | PASS |
| safety metric MANUAL (UNVERIFIED) | PASS |
| warp preview offers and labels the manual transform | PASS |
| saved to JSON | PASS |
| saved transform listed as Manual adjustment, MANUAL (UNVERIFIED), PREVIEW ONLY | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`ui_screenshots/I1_031-8_6_manual_adjustment.png`、`ui_screenshots/I2_031-8_6_saved_manual_record.png`
