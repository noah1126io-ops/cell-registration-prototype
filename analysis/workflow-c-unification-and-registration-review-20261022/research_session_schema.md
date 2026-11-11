# 研究ストアのスキーマ（`src/research_store.py`、schema v1）

## 1. 保存先と方式

- 保存先：`runs/research-store/research.sqlite`。環境変数 `CELLREG_RESEARCH_STORE` で変更できます。`runs/` はgit管理外です。
- NFS上にあるため、WALは使いません。
  - `journal_mode=DELETE`、`synchronous=FULL`。
  - 書き込みはすべて、ファイルロック（`research.lock`、`flock`）の中で `BEGIN IMMEDIATE … COMMIT` の1トランザクションとして行います。
  - DBに書くのはStreamlitのプロセスだけです。Slurmジョブはrunディレクトリにファイルを書き、UIの `sync_runs` が取り込みます。
- 追記のみ：`transforms`、`reviews`、`annotations`、`events` には UPDATE / DELETE を禁止するtriggerがあります。訂正は、`supersedes` で前の行を指す新しい行として保存します。
- `runs` は実行状態と結果の要約だけを `update_run` で更新します。artifactのパスは一度設定したら変えられません（`ValueError`）。
- スキーマ版は `meta.schema_version` に記録します。コードより新しい版のDBは開きません。
- 大きなartifactはJSONファイルのままにして、DBにはパスとsha256を保存します。

## 2. テーブル

| テーブル | 主な列 | 意味 |
|---|---|---|
| `meta` | key, value | `schema_version` |
| `sessions` | id, name, sample_id, note, created_at, created_by | 実験セッション。UIで選択し、保存はすべてこのセッションに紐づく |
| `runs` | id(=runディレクトリ名), session_id, sample_id, mode(automatic/assisted/imported), localization, fixed, moving, state, slurm_job_id, run_dir, artifact_path, artifact_sha256, status, accepted, selected_candidate, failure_stage, input_fingerprint, software_json, params_json, constraints_json, annotation_ids_json, created_at, created_by, completed_at | Registrationの実行 |
| `transforms` | id, session_id, sample_id, run_id, candidate_id, kind(automatic/candidate/manual/landmark/imported), parent_id, fixed, moving, M_json, t_json, status, label, note, detail_json, supersedes, created_at, created_by | 変換の版。manual / landmark の status は `MANUAL_UNVERIFIED` だけが許される |
| `reviews` | id, session_id, sample_id, run_id, candidate_id, transform_id, judgement, problems_json, comment, view_json, supersedes, created_at, created_by | 構造化された評価（`structured_feedback_schema.md`） |
| `annotations` | id, session_id, sample_id, kind(roi/search_roi/landmarks/exclusion/manual_init/note), frame, geometry_json, payload_json, label, run_id, candidate_id, transform_id, supersedes, created_at, created_by | 座標を持つ注釈 |
| `events` | id, ts, session_id, sample_id, actor, action, payload_json | 操作の履歴（session.create、run.register、run.update、transform.add、review.add、annotation.add） |

## 3. 保存されるもの（依頼の項目との対応）

| 依頼の項目 | 保存場所 |
|---|---|
| Session ID | `sessions.id`、各行の `session_id` |
| Run ID | `runs.id`（＝ `runs/registration-explorer/<id>`）、artifactの `run_id` |
| Sample ID | 各行の `sample_id`（サンプル間で混ざらない。クエリは常にsample_idで絞る） |
| Input data fingerprint | artifactの `input.{fixed,moving}`（核座標のsha256、点数、撮像範囲、spacing、画像のサイズ・mtime）、`runs.input_fingerprint` |
| Software version | artifactの `software`（core版、git HEAD、srcのdirty、python・numpy）、`runs.software_json` |
| Algorithm / Parameters | artifactの `config`（localization、image_feature、探索設定）、`runs.params_json` |
| Fixed / Moving、Coordinate frames | `runs.fixed/moving`、artifactの `frames`（Frame.to_dict、変換の規約） |
| Global search candidates | artifactの `localization`、`candidates[].search` |
| Refinement results | `candidates[].refine`、`flags` |
| Transform matrices | artifactの `transform.forward/inverse`、`candidates[].T_search/T_final`、`transforms.M_json/t_json` |
| Safety verdict | artifactの `decision`、`runs.status/accepted/failure_stage` |
| Image usability / fallback | artifactの `image_status`（flags、reasons、品質） |
| Manual transform | `transforms`（kind=manual / landmark、detailに基準の変換・調整量・UI診断） |
| Review annotations | `reviews` |
| ROI / landmarks | `annotations`（geometryは表示中のFixed frameの座標） |
| Timestamps | 各行の `created_at`（UTC）、`events.ts` |

## 4. エクスポート

- JSON（`cellreg-research-store-export/1`）：session、runs、transforms、reviews、annotations、events。`*_json` 列は展開した形で出力します。
- CSV：5つのテーブルをzipにまとめます。
- 範囲：「このセッション」または「このサンプルのすべて」。
