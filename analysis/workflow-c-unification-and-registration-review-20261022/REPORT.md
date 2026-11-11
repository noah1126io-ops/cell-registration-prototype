# Workflow Cの統合とRegistrationの研究環境（2026-10-22）

対応領域の探索 → 候補の幾何変換 → 核配置の精密化 → 信頼性の評価 → Web表示 → 利用者の評価・修正 → 保存 → 次の実験での再利用、までを1つの流れで動くようにしました。中心にあるのは、共通エンジン `src/registration_core`（研究Workflow C）です。

- branch：`feature/gpu-slurm`（HEAD `f66ecbe` + 未commitの変更）。commit・pushはしていません。
- 詳細：
  - `as_is_architecture.md`（変更前）
  - `to_be_architecture.md`（変更後）
  - `module_dependencies.md`
  - `global_search_diagnostics.md`
  - `workflow_parity_results.md`
  - `ui_implementation_report.md`
  - `research_session_schema.md`
  - `structured_feedback_schema.md`
  - `ui_test_results.md`
  - `known_limitations.md`

## 0. 結果の要約

| サンプル | 画像 | 画像探索（C1） | 核探索（C1） | 判定 | NN median / ≤3 µm |
|---|---|---|---|---|---|
| 031-6_2（HE Fixed） | 使える | UNIQUE（0.504） | UNIQUE（0.477） | **ACCEPTED**（両探索が同じ領域） | **1.9078 µm / 0.5995**（既存の結果と差 0） |
| 031-6_2（IF Fixed） | 使える | UNIQUE（0.427） | UNIQUE（0.478） | **ACCEPTED** | 1.8855 µm / 0.5985 |
| 031-8_6（HE / IF Fixed） | 使える | AMBIGUOUS（0.006 / 0.035） | AMBIGUOUS（0.031 / 0.037） | **未採択**、C1で停止 | 偶然一致と区別できない |
| 031-8_6（ROIで探索範囲を制限、assisted） | 使える | AMBIGUOUS（0.005） | AMBIGUOUS（0.023） | ASSISTED_REVIEW_REQUIRED（自動採択しない） | 偶然一致並み |

- Parity：analysisの研究コードと新しいエンジン、保存済みの結果、旧Runner、UI（Slurm Runner）経由と直接呼び出し。いずれも変換の差は **0** です。
- テスト：pytest **262 passed、1 skipped**。ブラウザE2E **96 / 96**（アプリの再起動をはさんだ保持の確認を含む）。
- 既存のanalysis（2,961ファイル）は変更していません。git管理下のファイルで変わったのは `README.md` と `app.py` だけです。

---

## A. システム構造

**1. Workflow Cと研究Runnerは、何が分断されていたか**

- Production Workflow Cのglobal stepは `estimate_affine_with_y_flip` です。30°刻みの初期値から全点でICPを行うもので、対応領域の探索ではありません。
- 対応領域を探す研究コード（A2、E0x、画像探索）は `analysis/` 配下にしかありませんでした。`sys.path` の書き換えと絶対パスで読み込んでいました。
- 採択判定は、実行後に別のスクリプトで行っていました（`aggregate.status` はIF frame、`e0xlab` はfixed frame）。経路によって判定がずれ得る状態でした。
- 結果の形式は6種類ありました。Explorerからは画像探索を実行できず、保存できるのは手動調整のJSONだけでした（`as_is_architecture.md`）。

**2. 何を共通化したか**

- A2・pose探索・重なり領域ICP・評価・構造mapを、関数本体を変えずに `src/registration_core` へ移しました。差 0 を確認しています。
- その上に、C1–C4のorchestration（`workflow.run_workflow`）を作りました。
  - 画像を使えるかの判定
  - 画像探索と核探索の両方の記録
  - 候補ごとの精密化と検証
  - 事前固定の規則による判定（CONFLICT、assistedを含む）
  - 共通artifact（`cellreg-registration-artifact/1`）
- 研究Runner、Explorer、Workflow Cの研究モード、検証スクリプトは、すべてこの1つの関数を呼びます。

**3. 現在の実行経路**

| 起点 | 経路 |
|---|---|
| Explorer『新規実行』 | `jobs.submit` → sbatch → `run_registration_explorer_job.run_engine` → `run_workflow` → `artifact.json` → `sync_runs` で研究ストアへ |
| Workflow C（研究モード） | STEP 2で採択済みの結果を明示的に選ぶ → `workflow_c_research.affine_from_research` → 従来のfine stage・ゲート・artifact |
| Workflow D | Workflow C result artifact（従来）。研究結果は `workflow_d_bridge` で同じ形式に書き出してから読み込む |

（`to_be_architecture.md`）

**4. Workflow Dとの接続**

- Workflow Dのコードは変えていません。
- 研究結果は、`check_handoff` を通ったもの（採択済み・完全自動・HE/IF）だけを、Workflow C result artifactとして書き出せます。affineのみで、fine stageは `off` です。
- 未採択の候補・注釈を使った結果・手動変換は、拒否します（UIとpytestで確認）。
- Workflow Cの画面で研究global affineを選んだ場合は、従来どおりのWorkflow C artifactができ、そのままDへ渡ります。

## B. Registration

**5. 画像がある場合、最初に組織領域を探索できるか**

- できます。`image_status` で「ファイルがある」と「使える」を分けて判定します（品質：組織の割合、自分の核と核mapのAUC）。
- 使える場合は、構造map（tissue＋核map。生の輝度は比べません）でpose探索を先に行います。
- 031-6_2では、画像だけで正しい領域をUNIQUEで見つけました（margin 0.504）。探索poseは2026-10-19のE2tnと差 0 です。
- 核探索も別に行い、両方の候補を保持します。

**6. 画像がない場合、Point-onlyへ正しく移行できるか**

できます。

- 画像を隠す：031-6_2で1.9078 µm、ACCEPTED
- 画像の読み込み失敗：`IMAGE_READ_FAILED` を記録して核探索へ移行し、ACCEPTED
- 画像も物理範囲もない旧ペア：A2とそのdetector。保存値と差 0

（`workflow_parity_results.md` §2）

**7. 031-6_2で既存結果と一致するか**

一致します。

- 核のみ：2026-10-19のE0x、2026-10-20の旧Runner X3と、変換の差 0。NN median 1.9078 µm、≤3 µm 0.5995。
- 自動（画像優先）：最終変換も同じ解です。
- UI（Slurm Runner）経由と直接呼び出し：全候補で差 0。

**8. 031-8_6で候補領域を実際に可視化できるか**

できます。

- 「対応領域の探索」タブで、Fixed全体に画像探索4候補・核探索6候補の領域を表示します（Test K、`screenshots/K1_*`）。クリックすると、その候補のPreview（PREVIEW ONLY）に切り替わります。
- 結果は、候補がH&E全体に散らばり、画像1位と核1位は2.3 mm離れていました。どの候補も核配置は偶然一致並みです。対応領域は**未解決**のままとしています（`global_search_diagnostics.md` §2）。

**9. Global LocalizationとICPの失敗を区別できるか**

できます。

- 探索ごとに、次を別々に記録しています。
  - C1のverdict
  - 探索poseの時点の核配置
  - ICPの移動量
  - `RELOCATED_BY_ICP` / `ICP_LEFT_TRUST_REGION`
  - 最終の核配置
- 最初に止まった段階をC1 / C2 / C3で表示します。
- 例：031-8_6はC1で停止しました（探索が一意に決まらない）。031-6_2の下位候補は、ICPが600 µm動かして重なりのない配置に縮退し、null比が1.0〜1.2でした。これはC2の問題として区別できます。

## C. Webアプリ

**10. UIの略語を分かりやすい名称に変えたか**

変えました（`ui_implementation_report.md` §2）。A2・E0x・E1〜E3・ICP・verdictを日本語の説明にしています。内部IDは詳細欄・artifact・ログに残しています。

**11. 手動で位置・回転・倍率を調整できるか**

できます。

- 回転・倍率・平行移動を調整できます。中心と刻みを選べ、自動推定へのリセットもあります。
- 自動推定との比較表と、自動推定の領域（白い破線）を並べて表示します。
- 対応点を2組以上クリックすると、相似変換と残差を求めます。
- 保存すると新しい変換（`MANUAL_UNVERIFIED`）になり、元の自動推定は上書きしません。

**12. 候補の正誤や問題の種類を選択できるか**

できます。正しい / 間違い / 判断できない、問題8種類、コメントを選べます。run・候補・変換・表示中の変換と一緒に保存します（Test N）。

**13. 実験セッションを保存・再読込できるか**

できます。SQLiteの研究ストアに保存します（追記のみ、schema v1、ファイルロック、1トランザクションで書き込み）。ブラウザの再読込とアプリの再起動の後も残ることを、Test N2で確認しました。

**14. 保存した注釈を次の実験で再利用できるか**

できます。

- ROI・除外・手動の初期位置・対応点を、『注釈を使った探索』で明示的に選ぶと `Constraints` になります。
- 031-8_6のROIで実行しました：ROI内の1,537 / 4,800核で探索し、ASSISTED_REVIEW_REQUIRED。
- 保存しただけでは自動実行に反映されません。assistedとして記録し、自動採択しません。

**15. 自動結果と手動結果を区別して表示できるか**

できます。

- 結果の種類が分かれています（研究Workflow C / 手動調整した変換（未検証））。
- バナー（MANUAL（未検証））、安全性の表示（MANUAL (UNVERIFIED)、ASSISTED_REVIEW_REQUIRED）、ストアの `kind` でも区別します。

## D. 品質・安全性

**16. 既存テストに影響はないか**

ありません。`pytest tests` で262 passed、1 skipped です（既存の248件はすべて通過、新規14件）。

**17. Browser E2Eは通過したか**

**96 / 96** で通過しました（`ui_test_results.md`）。テスト専用のサーバー（8512）とストアを使い、利用者の研究記録にはテストデータを入れていません。

**18. 保存データに破損や上書きの問題はないか**

- `transforms` / `reviews` / `annotations` / `events` は、triggerでUPDATE / DELETEを禁止しています。runのartifactは差し替えられません。
- `integrity_check` は ok です。
- 既存のanalysisはハッシュで変更なしを確認しました（2,961ファイル）。原画像は読むだけです。

**19. Productionの既定動作は維持されたか**

維持されています。

- 研究モードOFFでは変化がありません。
- 研究モードでも、Workflow Cの研究global affineは既定OFFで、採択済みの結果を明示的に選んだときだけ使われます。
- fine stage・安全ゲート・Workflow Dは変えていません。Workflow A〜Dのページも例外なく表示されます（Test F・R）。

**20. 未完成・未検証の機能**

`known_limitations.md` を参照してください。主なものは次のとおりです。

- 画像探索の設定は6_2を見た後に選んだもので、独立した検証はしていません。
- 031-8_6は未解決です。
- 画像探索は遅いです（HE Fixedで約25分、Runner経由で最大54分）。
- 注釈が探索をどれだけ改善するかは評価していません。
- 候補表の行選択はブラウザE2Eで自動確認していません（地図クリックでは確認済み）。
- 双方向viewerは、Streamlit 1.62の内部プロトコルに依存しています。
- Workflow Cの画面で研究Workflow C自体を実行することはできません（Slurm経由）。

---

## 1. 変更したファイル

**新規**

- `src/registration_core/`：`__init__.py`、`a2.py`、`search.py`、`pipeline.py`、`features.py`、`frames.py`、`metrics.py`（以上は研究コードの移設）、`workflow.py`、`workflow_d_bridge.py`
- `src/research_store.py`
- `src/workflow_c_research.py`
- `src/registration_explorer/research_ui.py`
- `src/registration_explorer/interactive/index.html`（`viewer.component_html()` が生成）
- `tests/test_registration_core.py`、`tests/test_research_store.py`

**変更**

- `app.py`（研究モードの欄と、affineの選択の2か所。+44 / −20）
- `README.md`（研究Workflow Cの説明）
- `src/registration_explorer/` の `page.py`、`viewer.py`、`results.py`、`scene.py`、`jobs.py`、`catalog.py`
- `scripts/run_registration_explorer_job.py`

**このディレクトリ**

- 設計・結果の文書：`as_is_architecture.md`、`to_be_architecture.md`、`module_dependencies.md`、`global_search_diagnostics.md`、`workflow_parity_results.md`、`ui_implementation_report.md`、`research_session_schema.md`、`structured_feedback_schema.md`、`ui_test_results.md`、`known_limitations.md`
- スクリプト：`run_core_checks.py`、`run_checks_srun.sh`、`compare_runner_vs_direct.py`、`ui_e2e.py`、`make_ui_test_results.py`、`test_server.sh`
- データ：`cache/`、`logs/`、`screenshots/`、`freeze/`、`e2e_state/`（テスト用のストア）

## 2. 起動・操作

- 開発サーバー：
  - 起動：`analysis/e0x-difficulty-and-web-integration-20261020/dev_server.sh start 8511`（停止は `stop`、確認は `status`）
  - コードを変えたら、`stop` → `start` で反映します。
  - 2026-10-22の時点で、最新のコードで起動しています。
- ブラウザ：SSHで転送（`ssh -N -L 8511:127.0.0.1:8511 <user>@<gateway>`）し、`http://localhost:8511` を開きます。サイドバーで **Registration Explorer (Research)** を選びます。
- 操作方法と保存先はREADMEの「Registration Explorer (Research) と研究Workflow C」にあります。
- 研究ストア：`runs/research-store/research.sqlite`。検証用のセッション「検証 2026-10-22（Claude、UI経路の自動実行）」に、031-6_2・031-8_6の自動実行が登録されています。
- Cloudflareトンネルの再公開はしていません。他の利用者のプロセスやジョブには触れていません。
