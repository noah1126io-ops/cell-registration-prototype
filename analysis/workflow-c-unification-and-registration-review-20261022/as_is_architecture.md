# As-Is：現在のシステム構造（2026-10-22の監査、変更前）

実コードを読んで確認した事実だけを書きます。行番号は2026-10-22時点（HEAD `f66ecbe` + 未commitの変更）のものです。

## 1. 全体図（変更前）

```text
                       ┌──────────────────────── app.py（Streamlit, 6187行）────────────────────────┐
                       │ Workflow A/B          Workflow C (show_he_geojson_preparation 1384–5895)     │
                       │                         STEP1 入力: HE核 .npy(px) + 蛍光GeoJSON(µm) [+HE画像]  │
                       │                         global: estimate_affine_with_y_flip ──────────────┐  │
                       │                         fine:   joint flow / density flow / off / …       │  │
                       │                         gate:   _density_flow_trial_is_safe 等            │  │
                       │                         出力:   workflow_c_registration_result.zip ──┐    │  │
                       │ Workflow D (5897)  ◄──── session / upload ─────────────────────────────┘    │  │
                       │ Registration Explorer (CELLREG_RESEARCH_MODE=1 のときのみ)                   │  │
                       └──────────────┬──────────────────────────────────────────────────────────────┘  │
                                      │ 保存済みartifactを読むだけ / Slurmへ1件投入                           │
          src/pointset_registration.py │  icp_similarity(177) / affine_icp(263) / estimate_affine_with_y_flip(331) ◄┘
                                      │
   ┌──────────────────────────────────┴───────────────── 研究コード（analysis/ 配下のみ）──────────────┐
   │ affine-global-initialization-20261014/init_lib.py  : A2 = method_a2(292), refine(125), detector(495), masked_ncc(267) │
   │ optional-image-assisted-registration-20261019/oir/ : search.PoseSearch, pipeline.run_method / cropped_icp / evaluate, │
   │                                                      features.extract, frames.Affine2D / TransformArtifact             │
   │                    run_real.py / aggregate.py      : 実行と採択判定（aggregate.status）、artifact JSONの作成          │
   │ e0x-difficulty-and-web-integration-20261020/e0xlab.py : X0–X3 ablation（Canvasをmonkeypatch）, run_variant         │
   └───────────────────────────────────────────────────────────────────────────────────────────────────┘
          ▲  sys.path.insert（絶対パス・相対パス）で読み込み
   scripts/run_registration_explorer_job.py（Explorerの研究用Runner, Slurm CPU）→ e0xlab.run_variant / init_lib.method_a2
```

## 2. 問いへの回答（監査結果）

| # | 項目 | 実装場所 | 内容 |
|---|---|---|---|
| 1 | Workflow Cの処理順序 | `app.py::show_he_geojson_preparation`（1384–5895） | 入力（1406–1475）→ 実行ゲート（2486–2502）→ global affine（2576–2594）→ 組織validity（2607–2669）→ fine stage（2670–2905）→ 変種と評価（2912–3160）→ artifact（5698）、D向けhandoff（5733）、run bundle（5770） |
| 2 | Global Search | Production：`src/pointset_registration.py::estimate_affine_with_y_flip`（331）。研究：`init_lib.method_a2`（A2）、`oir/search.py::PoseSearch`（E0x/E1/E2/E3） | Production：flip候補4種 × 0–330°の30°刻み初期値 → `icp_similarity` → `affine_icp`。**全点を使うICPで、対応領域の探索ではない**。研究：密度map/画像特徴mapのmasked NCC（FFT）で、回転×scale×並進を探索 |
| 3 | ICP | `src/pointset_registration.py::icp_similarity`（177）, `affine_icp`（263） | Productionは全点。重なり領域への制限は研究の `oir/pipeline.py::cropped_icp`（70）だけ |
| 4 | E0xの実行経路 | `oir/pipeline.py::run_method(method="E0x")` → `run_search` → `PoseSearch.run` → `cropped_icp`。ablationは `e0xlab.run_variant`。Explorerは `scripts/run_registration_explorer_job.py` → `e0xlab.run_variant` | 採択判定は実行後に別ファイルで：`aggregate.status`（IF frame）、`e0xlab.run_variant`（Fixed frame）。**2つの経路で判定のframeが違う** |
| 5 | 画像を使う手法 | `oir/features.extract`（tissue/nuclear/edt/envelope、`usable`判定）→ `pipeline.build_layers`（E1/E2/E3）→ `PoseSearch`（robust zの合成 zsum/zmin/rerank） | fallback（IMAGE_INCOMPATIBLE、IMAGE_REGISTRATION_UNCERTAIN → POINT_ONLY_FALLBACK）は `aggregate.status` のみ。`run_method` の中にはない |
| 6 | Web Explorerが呼ぶ処理 | `src/registration_explorer/results.py`（保存済みJSONの読み込みのみ）、`jobs.py::submit` → `slurm/registration_explorer_cpu.sbatch` → `scripts/run_registration_explorer_job.py` | 新規実行はX0–X3（E0xのablation）とA2だけ。画像を使う手法（E1–E3）は実行できない |
| 7 | Transformの保存・読み込み | 研究：`analysis/.../artifacts/<sample>_<config>.json`（`oir-transform-artifact/0.1`、aggregate.py 203–222）、`cache/real_*.json`、`cache/abl_*.json`、`runs/registration-explorer/<id>/result.json`、`runs/registration-explorer-manual/<sample>/*.json`。Production：`workflow_c_registration_result.zip`（`src/workflow_c_result.py`, `workflow-c-registration-result-v1`） | **形式が5種類**。研究側は `ResultRecord` に変換して表示（`results.py`） |
| 8 | UIでTransformを適用する場所 | `src/registration_explorer/page.py::_viewer_tab` → `results.oriented` → `scene.build_scene`（点は順写像、画像は画素→world affineに変換を合成）。Warpは `imaging.warp_preview`（逆写像） | |
| 9 | Workflow Dへ渡される情報 | `app.py::show_raster_deformation_workflow`（5897）、`src/workflow_d.py`（`affine_result_from_artifact` 40, `fine_result_from_artifact` 59） | Workflow Cのzipのみ。affine（HE px → 蛍光 µm、flip込み）、fine field、manifest（`fine_applied`など）。Dは再判定せず `fine_applied` に従う。**研究結果からDへの経路はない** |
| 10 | 重複と分断 | 下表 | |

## 3. 重複・分断箇所

| 分断 | 内容 | 影響 |
|---|---|---|
| Global探索が2系統 | Production（30°刻みICP初期値、全点）と研究（A2/E0x：密度NCC探索、重なりICP）が別実装 | 研究で有効だった対応領域探索が、Workflow Cからは使えない |
| 研究コードの置き場所 | A2/E0x/画像探索は `analysis/` 配下にしかなく、`sys.path` の書き換えと絶対パス（`oir/pipeline.py` 21–26、`oir/sample.py` 19–20、`init_lib.py` 29–38）に依存 | `src/` から安全にimportできない。`init_lib` は他のanalysisディレクトリをsys.pathの先頭に入れ、`config.py` などの名前が衝突し得る |
| 採択判定が実行後 | `run_method` は判定せず、`aggregate.status`（IF frame）と `e0xlab.run_variant`（Fixed frame）が別々に判定 | 同じ結果でも経路によって判定がずれ得る |
| Artifact形式 | 研究5種 + Production 1種 | Viewerは形式ごとのadapterが必要。候補・処理段階・入力のfingerprintを共通に保存する形式がない |
| 画像探索とPoint-only | 研究では別々の設定（E2とE0x）として並べて実行し、fallbackは集計スクリプトの中 | 「画像を優先し、使えなければ点群」という1つの処理として動いていない |
| UIからの実行 | ExplorerはX0–X3とA2のみ、画像探索は不可。Workflow Cとは別のRunner | 研究Runner・Workflow C・Explorerでアルゴリズムの呼び出しが別 |
| 保存 | Workflow Cのin-process実行は保存されない（download only）。Explorerは手動変換のJSONのみ。セッション・評価・注釈を保存する仕組みはない（DBなし） | 研究者の試行錯誤が残らない |

## 4. 既存の制約（変えてはいけないもの）

- `tests/test_workflow_c_ui.py`：`WORKFLOW_C_PRIMARY_FINE_METHODS` のtuple、既定 `index=0`、widget key、STEP 1–5、タブ名をソースから検査している。
- `tests/test_workflow_c_d_split.py`：DがCの関数（`estimate_affine_with_y_flip`、flow関数）を参照しないこと、`workflow_c_raster_outputs_enabled = False`。
- Production Workflow Cの座標規約：`p_world = A @ flip(p_px) + t`（HE px → 蛍光 µm、Aがpx→µmのscaleを含む）。artifactは2×2のAと長さ2のt。
- 研究の座標規約：各modalityのframe-local world（µm、原点UNKNOWN、`world = pixel × pixel_um`）、`fixed_world = M @ moving_world + t`。
- 採択規則（2026-10-19事前固定）：探索verdictがUNIQUE、かつ ≤3 µm ≥ 0.25、かつ 鏡像nullの3倍以上。
