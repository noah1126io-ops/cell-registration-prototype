# To-Be：統合後の構造（2026-10-22に実装）

## 1. 全体図

```text
                         ┌──────────── 共通エンジン src/registration_core/（研究Workflow C） ─────────────┐
                         │ run_workflow(sample, fixed: SideInput, moving: SideInput, WorkflowConfig, Constraints)│
                         │   C1 localize   image_status → 画像探索（構造map, 使えるときだけ）＋ 核探索（常に）   │
                         │                 / A2（従来法） / 注釈を使った局所探索                                  │
                         │   C2 refine     候補ごとに重なり領域ICP（cropped_icp）、移動量・RELOCATED を記録       │
                         │   C3 validate   事前固定の規則（UNIQUE ∧ ≤3µm≥0.25 ∧ null比≥3）、CONFLICT、ASSISTED    │
                         │   C4 artifact   cellreg-registration-artifact/1（候補・判定・入力fingerprint・版）     │
                         │ 中身: a2.py / search.py / pipeline.py / features.py / frames.py / metrics.py          │
                         │       （2026-10-14/19 の研究コードを関数本体そのままで移設。parity = 差 0）           │
                         └───────▲───────────────────▲─────────────────────▲──────────────────────▲──────────────┘
                                 │                   │                     │                      │
   scripts/run_registration_     │   Registration Explorer           Workflow C（app.py）      workflow_d_bridge
   explorer_job.py（Slurm）──────┘   src/registration_explorer/      研究モードでのみ、明示的に   採択済み・完全自動のみ
   config.engine=registration_core   ・catalog.side_input（入力）     選んだ採択済み結果を         → Workflow C result
   → artifact.json                   ・results.record_from_engine_   STEP 2 の global affine に   artifact（affineのみ）
                                       artifact（表示）               （src/workflow_c_research）  → Workflow D（無変更）
                                     ・research_ui / page（操作）
                                            │
                                            ▼
                         研究ストア src/research_store.py（SQLite, runs/research-store/research.sqlite）
                         sessions / runs / transforms / reviews / annotations / events（追記のみ）
                                            │  constraints_from_annotations（明示的に選んだ注釈だけ）
                                            └──────────────► Constraints → run_workflow（assisted）
```

## 2. 実行経路（統合後）

| 呼び出し元 | 経路 | 備考 |
|---|---|---|
| Explorerの『新規実行』 | `jobs.submit(cfg{engine})` → sbatch `registration_explorer_cpu.sbatch` → `run_registration_explorer_job.py::run_engine` → `registration_core.run_workflow` → `runs/registration-explorer/<id>/artifact.json` | 研究ストアにrunを登録し、完了時に `sync_runs` が要約と自動変換を記録 |
| 解析・検証スクリプト | `run_core_checks.py` → `run_workflow` | 同じ関数。parityの確認に使用 |
| Workflow C（Production画面） | 既定は従来どおり `estimate_affine_with_y_flip`。研究モードかつ利用者が採択済み結果を選んだときだけ `workflow_c_research.affine_from_research` | fine stage・安全ゲート・artifact・Dは無変更 |
| Workflow D | 従来どおりWorkflow C result artifactのみ。研究結果は `workflow_d_bridge.to_workflow_c_artifact`（採択済み・完全自動のみ）で同じ形式にしてから読み込む | Dのコードは無変更 |
| 旧Runner（X0–X3, A2） | 従来どおり（2026-10-20の結果を再現するため残す）。UIでは折りたたみに移動 | |

## 3. 段階の分離（C1–C4）

| 段階 | 入力 | 出力（artifactの場所） | 失敗の表示 |
|---|---|---|---|
| C1 対応領域の探索 | 両側の核座標・撮像範囲、使える場合は構造map（tissue・核） | `localization.{image,points,a2,init}`（verdict、margin、pose数、時間、使った層）、`candidates[].search` | NO_CANDIDATE / AMBIGUOUS / LOW_PEAK →「C1」 |
| C2 精密化 | 各候補のsimilarity pose | `candidates[].refine`（ICPの移動量、使った核数）、`flags`（ICP_NOT_RUN、RELOCATED_BY_ICP、ICP_LEFT_TRUST_REGION） | ICP不能・別領域へ移動 →「C2」 |
| C3 検証 | 最終変換 | `candidates[].validation`（fixed frame・IF frame・探索pose時点・重なり・画像の整合・対応点残差）、`decision` | 核配置が偶然一致並み・探索どうしの矛盾 →「C3」 |
| C4 記録 | — | `schema`、`input`（fingerprint）、`software`、`frames`、`config`、`constraints`、`transform`（forward / inverse） | — |

「探索poseの時点で核配置が合うか」（`search_pose_geometry_ok`）と最終の核配置を別に記録しているので、「探索が間違った領域を選んだ（C1）」と「ICPが悪化させた（C2）」を区別できます。

## 4. 画像の使い方

1. 「画像ファイルがある」と「位置合わせに使える」を分けて判定します（`image_status`）。
   - ファイルなし → `NO_IMAGE`
   - 読めない → `IMAGE_READ_FAILED`
   - 片側だけ → `ONE_IMAGE_ONLY` / `IMAGE_PAIR_UNAVAILABLE`
   - 品質不足 → `IMAGE_INCOMPATIBLE`（組織の割合、自分の核と核mapのAUC > 0.7）
   - 使える → `IMAGE_USABLE`
2. 使える場合は、画像探索（tissue＋核map、robust zのzsum）を先に行います。核探索は常に別に行い、候補は別々に保持します。どちらの候補も失われません。
3. 判定：
   - 画像だけが基準を満たす → ACCEPTED（画像主導）
   - 核だけが満たす → POINT_ONLY_FALLBACK（採択）
   - 両方が同じ領域 → ACCEPTED
   - 両方が別の領域 → CONFLICT（不採択）
4. 生の輝度は比較しません。H&Eは光学濃度とヘマトキシリン、蛍光はチャンネル最大値とB−R（Hoechst推定）を使い、どちらも同じ「組織の占有」「核の確率」に写してから比べます。

## 5. 安全性の境界

- 採択規則は2026-10-19の事前固定のまま変えていません。fixed frameに加えて、HE/IFの組ではIF frameでも満たす必要があります。これは従来の2経路（`aggregate.status` と `e0xlab`）のどちらよりも緩くありません。
- 注釈（制約）を使った実行は `mode=assisted`、`ASSISTED_REVIEW_REQUIRED` で、自動採択しません。
- 手動変換・対応点変換は、ストアで `MANUAL_UNVERIFIED` 以外の状態を持てません（コードで拒否）。
- Workflow Dへの受け渡し（ファイル書き出し）とWorkflow Cのglobal置換は、`check_handoff` を通った結果（採択済み・完全自動・HE/IF）だけです。
