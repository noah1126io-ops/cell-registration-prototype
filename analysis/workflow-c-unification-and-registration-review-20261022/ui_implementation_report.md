# Web UI（Registration Explorer）の実装報告（2026-10-22）

既存のRegistration Explorerを残したまま、対応領域の探索・検証・修正・保存を行う研究環境に作り替えました。研究モード（`CELLREG_RESEARCH_MODE=1`）のときだけ表示されます。

## 1. 画面の構成

| 場所 | 内容 |
|---|---|
| 上部 | Sample、Fixed/Moving、画像を使わない表示、画像読み込み失敗の再現。**実験セッション**（保存先の選択と新規作成。最後に使ったセッションを自動で復元） |
| 結果の選択 | 「結果の種類」は日本語の名称です（下表）。内部IDは結果の下に小さく表示します。安全性の表示には説明文が付きます（例：「曖昧：上位の候補どうしの差が小さく、対応領域が1つに決まらない（未採択）」） |
| タブ① 対応領域の探索 | 段階の診断（探索ごとのC1の判定と意味、核配置、この探索での採択、最初に止まった段階）、画像の利用状況、画像探索1位と核探索1位の差、候補一覧（行を選ぶとViewerに反映）、**候補マップ**（Fixed全体に全候補の領域を探索方法ごとの色で表示。クリックで選択） |
| タブ② Viewer | 重ね合わせ・Before/After・Zoom/Pan・核・推定対応・FOV・組織・候補領域を表示します。**クリック操作**（対応点・ROI・候補選択）、手動調整（自動推定との比較表、白い破線 = 自動推定の領域）、表示中の配置を注釈として保存（除外 / 初期位置） |
| タブ③ 評価・注釈 | 構造化された評価フォーム（正しい / 間違い / 判断できない、問題の種類8つ、コメント）、評価の一覧、注釈の一覧 |
| タブ④ 方式比較・⑤ Warp Preview | 従来どおりです。Warp Previewは手動調整中の変換も使えます |
| タブ⑥ 新規実行 | 研究Workflow C（自動・核のみ・従来法A2・注釈を使った探索）。処理段階のログ（C1 → C4）を表示します。旧Runner（X0–X3）は折りたたみの中です |
| タブ⑦ 実験記録 | 実行（判定・選ばれた候補・止まった段階・NN・≤3 µm・artifactのsha256）、変換、評価、注釈、操作履歴、JSON / CSVのエクスポート、DBのintegrity check |
| タブ⑧ Artifact | artifactの内容を表示します。採択済み・完全自動の結果だけ、Workflow D用（Workflow C result artifact、affineのみ）を書き出せます |

## 2. 表示名（内部ID → 画面）

| 内部 | 画面 |
|---|---|
| WF（registration_core） | 研究Workflow C：対応領域の探索 → 精密化 → 検証 |
| A2 / E0 | 核密度による対応領域探索（従来法） |
| E0x | 視野サイズの違いに対応した核探索 |
| E1 | 片側画像を使った補助解析 |
| E2 | 両画像からの対応領域探索 |
| E3 | 画像と核配置の統合探索 |
| ICP（cropped） | 重なり領域での核配置の精密位置合わせ |
| X0–X3 | 核探索の構成要素の比較（2026-10-20 ablation） |
| UNIQUE / AMBIGUOUS / LOW_PEAK / NO_CANDIDATE / LOCAL | 一意 / 曖昧 / ピークが弱い / 候補なし / 局所探索（それぞれ説明文付き） |
| ACCEPTED / POINT_ONLY_FALLBACK / CONFLICT / NOT_ACCEPTED / ASSISTED_REVIEW_REQUIRED / MANUAL (UNVERIFIED) | 説明文付きで表示（`results.SAFETY_TEXT`） |

## 3. 双方向viewer（クリックした座標をPythonへ返す）

- `viewer.interactive()`：Streamlitのcustom componentです。postMessageのプロトコルを直接使い、npmのbuildはありません。描画は従来のviewerと同じ `boot(S, API)` です。
- クリック操作：
  - **対応点**：Fixed側をクリック → 重ねて表示中のMoving側をクリック。番号付きの線で表示します。
  - **ROI**：ドラッグで矩形を描きます。
  - **候補選択**：候補領域の内側をクリックします。領域が重なる場合は、中心が近い候補を選びます。
- 座標：イベントはすべて表示中のFixed frame（µm）で返します。Moving座標は、Python側で**表示中の変換の逆写像**から求めます。各イベントには一意のnonceがあり、1回だけ処理します。
- Zoom・Pan・表示モード・透明度は、sessionStorageでStreamlitの再実行をまたいで保ちます（候補の切り替えや手動調整のたびに表示が戻らない）。

## 4. 保存（研究ストア）

- 保存はすべてSQLiteの研究ストアに行います（`research_session_schema.md`）。手動調整の保存も、2026-10-22からJSONファイルではなくストア（`transforms.kind=manual`）です。元の自動推定の行は変更しません。
- 評価は `reviews`、ROI・対応点・除外・初期位置は `annotations` に保存します（`structured_feedback_schema.md`）。
- Slurmの研究ジョブはrunディレクトリに書き、画面を開いたときの `sync_runs` が、状態・判定・artifactのsha256・自動変換をストアに取り込みます。

## 5. 確認中に見つけて直した不具合

| 不具合 | 発見 | 修正 |
|---|---|---|
| 候補polygonの形式を広げた後、boundsの計算が2要素tupleを前提にしていて例外になった | ブラウザでのスモークテスト | `scene.build_scene` で長さに依存しない扱いにした |
| セッションがない状態で初めて保存すると、表示済みのselectboxのkeyを変更してStreamlitの例外になった | ブラウザE2E（空のテスト用ストア） | 新しいセッションIDを保留し、次の実行の冒頭（widgetを作る前）に反映 |
| 候補領域が重なると、クリックで別の候補が選ばれた（面積最小の規則） | ブラウザE2E（Test K） | 中心が最も近い候補を選ぶように変更 |
| 評価・注釈の一覧がcanvas描画の表で、文字を読み取れなかった | ブラウザE2E | 行数が少ないので、通常の表（`st.table`）に変更 |
| `executescript()` が開いていたトランザクションを暗黙にcommitし、後のCOMMIT / ROLLBACKが失敗した | 単体テスト | DDLを1文ずつ実行。ROLLBACKはトランザクション中のときだけ |
| 画像ファイルが見つからないとき、「片側の画像しかない」とだけ記録していた | 画像失敗の実行（`wf_imgfail`） | `IMAGE_READ_FAILED` とファイル名を記録 |

## 6. 既存の動作への影響

- 研究モードがOFFのときは、Workflow選択の項目も含めて変わりません。
- 研究モードのWorkflow Cは、STEP 2に折りたたみの欄が1つ増えるだけで、既定はOFFです（Test R）。Workflow A〜Dのページは例外なく表示されます（Test F）。
- 旧Runner・保存済みの結果・手動調整の過去のJSONは、引き続き読み込んで表示します。
