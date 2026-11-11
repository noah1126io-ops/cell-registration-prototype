# UIテスト結果（2026-10-22）

- **ブラウザE2E**（headless Chromium + Playwright、`ui_e2e.py`。テスト専用サーバー http://127.0.0.1:8512 と、専用の研究ストア `e2e_state/research_e2e.sqlite` を使用）：**96 / 96 通過**
- **新しい単体テスト**（`tests/test_registration_core.py`、`tests/test_research_store.py`、`tests/test_registration_explorer.py`）：24 passed + データを使う2件 passed（2026-10-22）
- **テスト一式**（`pytest tests`、`--gres=gpu:1`）：262 passed、1 skipped（2026-10-22）

- 手順：A〜N を実行 → テストサーバーを再起動（アプリ再起動）→ N2〜S を実行。N2で、再起動前に保存した評価・ROIが残っていることを確認しています。
- 判定は、画面に表示された文字、viewerの状態（`window.__regx`）、ダウンロードしたファイルだけで行っています。
- 座標の確認（L・S）は、提供者のGeoJSONから直接読んだ核の座標と、artifactの変換行列を使い、アプリのコードとは独立に計算しています。

## Test A：031-6_2：既存結果の表示（Before/After・重ね合わせ・Warp）

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

スクリーンショット：`screenshots/A1_031-6_2_E0x_after_overlay.png`、`screenshots/A2_031-6_2_E0x_zoom_checker.png`、`screenshots/A3_031-6_2_E0x_vectors.png`、`screenshots/A4_031-6_2_before.png`、`screenshots/A5_031-6_2_warp_preview.png`

## Test B：031-8_6：AMBIGUOUS・候補（PREVIEW ONLY）

| 確認項目 | 結果 |
|---|---|
| AMBIGUOUS shown | PASS |
| PREVIEW ONLY + not passed to Workflow D | PASS |
| no Workflow D hand-off control | PASS |
| FOV of both sides + candidate regions drawn | PASS |
| candidate switch -> preview-only banner | PASS |
| candidate list (map) with IDs; candidates are PREVIEW ONLY | PASS |
| ablation X3 candidates listed (with ICP-refined regions) | PASS |
| ablation candidate preview banner | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/B1_031-8_6_E0x_top_candidate_fov.png`、`screenshots/B2_031-8_6_candidate1_preview.png`、`screenshots/B3_031-8_6_candidate_table.png`、`screenshots/B4_031-8_6_X3_candidates_with_icp.png`、`screenshots/B5_031-8_6_X3_candidate2_icp_preview.png`

## Test C：Point-onlyサンプル・画像を隠した表示

| 確認項目 | 結果 |
|---|---|
| point-only sample: images absent, nuclei present | PASS |
| image methods disabled with reason | PASS |
| points shown without images | PASS |
| H&E frame shown in px (spacing UNKNOWN), unit mismatch noted | PASS |
| transform shown | PASS |
| 031-6_2 point-only view: E0x still available | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/C1_point_only_pair_A2.png`、`screenshots/C2_031-6_2_images_hidden.png`

## Test D：画像読み込み失敗とfallback

| 確認項目 | 結果 |
|---|---|
| image methods unusable -> point-only result kept | PASS |
| E2 saved POINT_ONLY_FALLBACK with reason | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/D1_simulated_image_failure_alert.png`、`screenshots/D2_E2_point_only_fallback.png`

## Test E：Fixed/Movingの交換・逆Warp

| 確認項目 | 結果 |
|---|---|
| IF-fixed result selected by default | PASS |
| viewer frame is IF | PASS |
| swap consistency table (E0x 0.23 µm) | PASS |
| inverse warp HE->IF recorded | PASS |
| IF-fixed warp uses stored HE->IF map of E0x_IFfix (not inverted) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/E1_031-6_2_IF_fixed.png`、`screenshots/E2_swap_consistency.png`、`screenshots/E3_warp_HE_into_IF.png`

## Test F：既存Workflow A〜Dのページ

| 確認項目 | 結果 |
|---|---|
| Workflow A page renders without exception | PASS |
| Workflow B page renders without exception | PASS |
| Workflow C page renders without exception | PASS |
| Workflow D page renders without exception | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/F1_workflow_c_unchanged.png`

## Test H：旧Runnerの結果の表示

| 確認項目 | 結果 |
|---|---|
| completed explorer run listed with A2 detector verdict | PASS |
| IF-frame ≤3 µm of the new A2 run equals the saved Legacy A2 value (0.0532) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/H1_explorer_run_A2_result.png`

## Test I：手動調整（回転・倍率・平行移動）とストアへの保存

| 確認項目 | 結果 |
|---|---|
| manual mode shows MANUAL / unverified / not passed to Workflow D | PASS |
| viewer banner MANUAL | PASS |
| identity adjustment keeps base transform | PASS |
| rotation +2° and scale x1.02 applied to the displayed transform | PASS |
| zoom kept across reruns while adjusting | PASS |
| safety metric MANUAL (UNVERIFIED) | PASS |
| warp preview offers and labels the manual transform | PASS |
| saved to the research store (append-only) | PASS |
| saved transform listed as Manual adjustment, MANUAL (UNVERIFIED), PREVIEW ONLY | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/I1_031-8_6_manual_adjustment.png`、`screenshots/I2_031-8_6_saved_manual_record.png`

## Test J：研究Workflow C：段階の診断（C1/C2/C3）と画像探索・核探索の比較

| 確認項目 | 結果 |
|---|---|
| engine run listed (newest first) and decision shown | PASS |
| both searches recorded separately (image + points) | PASS |
| image-vs-point top candidate difference reported | PASS |
| internal ids moved to details | PASS |
| 031-8_6 not accepted, failure stage shown (C1) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/J1_031-6_2_stage_panel.png`、`screenshots/J2_031-8_6_stage_panel.png`

## Test K：候補マップと、クリックによる候補の選択

| 確認項目 | 結果 |
|---|---|
| map shows every candidate of every source | PASS |
| click on points#3 selects it in the Viewer | PASS |
| viewer shows the picked candidate as PREVIEW ONLY | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/K1_031-8_6_candidate_map.png`、`screenshots/K2_031-8_6_picked_candidate_preview.png`

## Test L：対応点のクリックと座標の定量確認・注釈として保存

| 確認項目 | 結果 |
|---|---|
| landmark table shown | PASS |
| landmark table present | PASS |
| clicked fixed / back-mapped moving coordinates match the nuclei (tol 6.87 um) | PASS |
| landmarks saved as annotation | PASS |
| similarity from landmarks shown with residual | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/L1_031-6_2_landmarks.png`

## Test M：ROIのドラッグと保存

| 確認項目 | 結果 |
|---|---|
| dragged ROI arrives in fixed-frame um | PASS |
| ROI saved as search-restriction annotation | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/M1_031-8_6_roi.png`

## Test N：構造化された評価の保存

| 確認項目 | 結果 |
|---|---|
| review saved as Review Annotation | PASS |
| review listed with judgement / problems / comment | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/N1_review_saved.png`

## Test N2：ブラウザ再読込・アプリ再起動の後の保持

| 確認項目 | 結果 |
|---|---|
| review persisted (new browser page) | PASS |
| annotations persisted (ROI) | PASS |
| no Streamlit exception on page | PASS |

## Test O：実験記録とJSON / CSVのエクスポート

| 確認項目 | 結果 |
|---|---|
| JSON export has reviews / annotations / events for this sample only | PASS |
| CSV export (zip of 5 tables) | PASS |
| store integrity ok | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/O1_experiment_record.png`

## Test P：注釈を使った探索の設定（明示的な選択）

| 確認項目 | 結果 |
|---|---|
| assisted run needs explicit annotation selection | PASS |
| selected ROI annotation accepted as a constraint (no validation error) | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/P1_assisted_run_setup.png`

## Test Q：Workflow Dへの受け渡し（採択済みのみ）

| 確認項目 | 結果 |
|---|---|
| accepted run offers Workflow D hand-off | PASS |
| download is a Workflow C result zip | PASS |
| unaccepted run refused for Workflow D | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/Q1_workflow_d_handoff.png`

## Test R：Workflow C（Production画面）の研究モードの接続

| 確認項目 | 結果 |
|---|---|
| research global option is off by default | PASS |
| only accepted automatic runs listed | PASS |
| no exception on Workflow C page | PASS |
| no Streamlit exception on page | PASS |

スクリーンショット：`screenshots/R1_workflow_c_research_global.png`

## Test S：表示座標の定量確認（画像と核）

| 確認項目 | 結果 |
|---|---|
| moving nuclei drawn at T(provider nuclei): max |d| 0.0099 um | PASS |
| moving image pixel of a nucleus lands on the drawn nucleus: max |d| 0.000 um | PASS |
| no Streamlit exception on page | PASS |
