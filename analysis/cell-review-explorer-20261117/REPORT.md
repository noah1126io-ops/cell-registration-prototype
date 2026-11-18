# Registration Explorer：細胞対応のレビュー → 点群の変形 → H&E 画像の変形（研究用、2026-11-17）

ブランチは `feature/gpu-slurm`、作業前の HEAD は `e8a68f6`。本番の既定動作は変えていない。

新しいタブ「細胞対応レビュー（研究）」は既定で無効で、`CELLREG_CELL_REVIEW=1` を付けたときだけ有効になる。

推定した対応も手動で修正した対応も、**生物学的に同じ細胞であることの証明ではない**。真の細胞対応のデータはなく、今後も用意しない前提で作った。

関連する文書：
- [audit.md](audit.md)：コードを変える前に行った監査（実装済みの処理と不足している処理）
- [test_results.md](test_results.md)：テストの結果
- [baseline_manifest.json](baseline_manifest.json)：作業前の sha256

## 1. 変更・追加したファイル

| 種類 | ファイル | 内容 |
|---|---|---|
| 追加 | `src/registration_research/cell_review.py` | 計算の処理（UI なし）：ROI の核対応、レビューの記録、アンカー、変位場、逆写像、画像の再標本化、出力 |
| 追加 | `src/registration_explorer/cell_review_ui.py` | 新しいタブ（描画と操作の受け渡しだけ） |
| 追加 | `src/registration_explorer/cell_review_viewer.py` | 核をクリックで選ぶビューアー。既存の `viewer.py` の描画テンプレートを import して、クリックの処理だけを入れ替えた。`viewer.py` は変更していない |
| 変更 | `src/registration_explorer/page.py` | タブの差し込み口だけ（+10 行）。無効のときは既存のコードの流れのまま |
| 変更 | `src/registration_research/cell_correspondence.py` | `run_pipeline` の返り値に `runs` を追加（1 行）。既存のキーと値は不変 |
| 追加 | `analysis/cell-review-explorer-20261117/` | 監査、テスト（`tests_crx`）、実データでの実行（`smoke_real.py`）、図、sbatch |
| 変更 | `README.md`、`docs/RESEARCH_STATUS.md` | 環境変数の表と、引き継ぎの文書 |

## 2. 再利用した既存の機能
- **Phase 1（対応領域）**：既存のものをそのまま使った。
  - engine の C1（画像あり／片側画像／点群だけの探索）
  - 「対応領域の探索」タブ（候補）
  - 「Viewer」タブの手動の回転・倍率・平行移動、ランドマーク、ROI の指定
  - 新しいタブでは、Viewer で表示中の変換（`regx-ctx`）、結果の最終変換、各候補から選び、H&E→IF の向きにそろえて使う。
- **Phase 2（核対応）**：凍結済みの B0 / B1 / B3 ＋ B2（10-14）。`cell_correspondence.run_pipeline` を通して使う。
  - 候補の順位は、B1 の UOT の `ranked_indices`（輸送量の順）。
  - 輸送量の割合は `UOTRun.plan`、graph support とコストは 10-08 の context から取る。
- **Phase 4（変形）**：
  - linear RBF（`deformation_experiment._rbf_from_anchors` → `validation_lib.fit_linear_rbf`）
  - UOT conservative（`validation_lib.run_uot`）
  - 安全性の監査 `common_audit`（SAFETY の閾値は変更なし）
  - `MIN_ANCHORS` = 20
- **Phase 5（画像）**：
  - ピラミッドの ROI 読み込み（`imaging.read_region`。触れるタイルだけを復号する）
  - 画素座標の規約（`level_corner_um`）
  - 表示用の変換（`to_display_rgb`）
  - 逆写像の解き方：本番の `pointset_registration._inverse_fine_warp` と同じ固定点反復。ただし残差のチェックを加えた。
  - 変位場の標本化は `registration_evaluation.sample_displacement_field`。
- **Viewer**：`viewer._TEMPLATE`、`points_layer`、`poly_layer`、`vectors_layer`、`scene.build_scene`（画像のレイヤー）。

## 3. 動作する画面と操作の手順
1. `CELLREG_CELL_REVIEW=1` を付けて Explorer を起動する。
   - 開発サーバーは、`CELLREG_CELL_REVIEW=1 analysis/e0x-difficulty-and-web-integration-20261020/dev_server.sh start 8511` で起動できる。2026-11-17 に、この設定で起動した。
2. サンプルと結果を選び、必要なら「Viewer」タブで候補を選ぶか、手動で調整する。
3. 「細胞対応レビュー（研究）」タブを開く。
   - **1. Global 変換と ROI**：使う変換を選ぶ（Viewer で表示中の変換、最終変換、候補）。ROI を「重なり全体」「Viewer で描いた ROI」「数値で指定」から選ぶ。**「自動推定を計算して保存」** を押す（6_2 の全体で約 2 分）。
   - **2. 核対応とレビュー**：
     - ビューアーで「H&E 核を選択」にして核をクリックすると、その核の情報と上位 5 候補が表示される。
       - 表示される情報：元の ID、座標（変換前と変換後）、周辺の核の数、B2 の状態とスコア、採用の根拠
       - 候補ごとの値：順位、距離、輸送量の割合、graph support、コスト
     - **支持する**、**除外する**、**未確定にする**、**候補に変更**ができる。「対応先の IF 核を選択」で任意の IF 核をクリックすると、**その IF 核を対応先に変更、または新しいペアとして追加**できる。
     - 核は ID を入力しても選べる。
     - **取り消す／やり直す**、**レビューを保存（新しい版）**。
     - 衝突（同じ IF 核に複数の H&E 核）は、警告と赤い線で表示する。一覧で状態ごとに絞り込めて、変更履歴も見られる。
   - **3. 点群の局所変形**：「変位場を計算／再計算」を押す。
     - 4 つの方式（Affine のみ、自動推定＋RBF、手動修正後＋RBF、UOT conservative）について、適用したかどうか、理由、アンカーの数と被覆率、Jacobian、Foldover、最大変位、p95 変位を表で比べる。
     - レビューを変えると、再計算を促す警告が出る。
     - 手動修正による点の移動量（中央値、95%、最大）を表示する。
     - 変形後の点群と、変位ベクトルを表示する。
   - **4. 画像と出力**：
     - 「画像を変形する」を押すと、Before / Affine / Nonrigid の H&E 画像を作る。表示では、IF の ROI 画像と、同じ変換をかけた H&E 核を重ねる。
     - 非剛体の変形が監査を通らなかった場合や、逆写像が収束しなかった場合は、Nonrigid の画像を表示しない。
     - 「結果をファイルに出力」を押すと出力する。ZIP でダウンロードもできる。

## 4. データの流れ（方向は H&E（moving）→ IF（fixed）に固定）
```
HE 核（HE frame）─[Global affine A：H&E→IF。Viewer / 候補 / 手動]→ IF frame
   → ROI（IF frame）の HE 核と、ROI±30 µm の IF 核
   → 凍結済み Graph / UOT / B2 → 自動推定（runs/cell_review/<sample>/auto/<auto_id>/。一度だけ書き込み、sha256 を記録）
   → レビュー（reviews/<review_id>/v0001.json, v0002.json, …。操作の記録と履歴。自動推定のファイルは読むだけ）
   → アンカー：自動推定 = B2 の matched、手動修正後 = 自動推定にレビューの操作を順に適用したもの
   → 変位場 d（IF frame の 4 µm 格子。linear RBF または UOT conservative）
     → common_audit ＋ アンカー数の確認 ＋ アンカーの被覆率の確認。通らなければ d = 0（Affine のみ）
   → 点：  x_IF = A(x_HE) + d(A(x_HE))
     d は格子を bilinear で標本化したもの。格子の外では、境界の値が 50 µm かけて 0 になる。
   → 画像：IF の画素中心 p → q + d(q) = p を固定点反復で解き、残差 ≤ 0.05 µm を確認 → x_HE = A⁻¹(q)
           → H&E のピラミッドの level から bilinear で標本化（ROI のタイルだけ）
   → 出力（exports/<id>/）：
     auto_correspondence.csv、auto_pairs.csv、reviewed_pairs.csv、nuclei_coordinates.csv（変換前、Affine 後、各方式の変形後）、
     global_transform.json、displacement_fields.npz、safety.json、review_effect.json、review_history.json、settings.json、*.png
```
- 研究ストア（research.sqlite）には書き込んでいない。Workflow D にも送っていない。
- 出力は `runs/`（git の対象外）に置く。

## 5. テストの結果（詳細は test_results.md）
- 既存のテスト：`tests/` 270 passed / 1 skipped、凍結済み 10 ＋ 7 passed、11-13 の 13 passed。どれも作業前と同じ。
- 新しいテスト `tests_crx`：23 passed、1 skipped（計算ノードに `node` がないため。ゲートウェイでは passed）。
- 仕様の Phase 7 で求められた 9 項目は、すべてテストで確かめた。

## 6. 実データでどこまで動いたか（`smoke_real.py`、Global は 11-11 のベンチマークの最終変換を逆向きにしたもの）

| | 031-6_2（切片関係は CONFLICT） | 031-8_6（隣接切片） |
|---|---|---|
| ROI 内の核 | HE 827 / IF 1091 | HE 1099 / IF 1552 |
| 自動推定（B2） | matched 612、ambiguous 95、unmatched 120。衝突 4 件 | matched 8、ambiguous 103、unmatched 988 |
| 自動推定＋RBF | 適用（アンカー 612、Jacobian の最小値 0.71、最大変位 5.0 µm） | 不適用（`ANCHORS_INSUFFICIENT`、8 組） |
| 手動修正後＋RBF（機械的な操作 9 件） | 適用（アンカー 610）。1 µm 以上動いた核 32 個、最大 5.5 µm | 不適用（6 組） |
| UOT conservative | 適用（最大変位 3.0 µm） | 適用（最大変位 4.7 µm） |
| H&E 画像の非剛体変形（2 µm/px、ROI 全体） | 適用。逆写像の残差 < 1e-12 µm | D で適用。手動修正後は不適用（Affine のみ） |
| 計算時間とメモリ | 自動推定 114 秒、変位場 6 秒、画像 3 枚 3 秒。peak 151 MB | 154 秒、3 秒、3 秒。peak 236 MB |

- 図 [figures/fig1_031-6_2.png](figures/fig1_031-6_2.png)：Affine の画像でも Nonrigid の画像でも、同じ変換をかけた H&E 核が、画像上のヘマトキシリンの核に重なる（座標の扱いが一致している）。
- **ここでの手動の修正は、スクリプトによる機械的な操作で、人が判断したものではない**（例：最も長い自動推定の対応 5 組を除外）。実データを人がレビューした結果は、まだない。
- 8_6 は隣接切片なので、核の一対一対応は期待できない。B2 の matched が 8 件しかないのは、それと整合する。D が「適用」になっても、細胞が同じであることや、位置合わせが良くなったことを意味しない（11-13 で、8_6 のランドマークの残差は改善しなかった）。
- 6_2 は切片関係が CONFLICT のまま（RESEARCH_STATUS §4）。matched 612 件も、細胞が同じであることの証拠ではない。
- serial_section_pair（点群のみ）は、AppTest で画面の流れが動くことだけを確かめた（ROI は 320 µm 四方）。

## 7. 未実装・未検証のこと
1. **実際のブラウザでのクリック操作は、自動では確認していない**。ヘッドレスブラウザの導入が、このセッションの権限で拒否されたため。代わりに、JS を Node で単体テストし、Streamlit 側は AppTest で確認した。開発サーバー 8511 でのご確認をお願いします。
2. **人による実際のレビュー**と、その効果の評価（独立のランドマークで TRE がどう変わるか）はしていない。真の対応がないので、精度の主張はしない。
3. 自動推定は同期的に計算するので、大きな ROI では画面が数分止まる。Slurm での非同期の実行は未実装。
4. 衝突は検出して表示するが、自動では解消しない。アンカーから除くかどうかは選択式（既定は「除かない」。11-13 の C と同じにするため）。
5. 本番の H&E の非剛体の逆写像（`warp_he_image_to_world`、8 回で打ち切り、収束チェックなし）は変更していない。研究用の側にだけ、収束チェックを入れた。
6. 新しく加えたゲートが 2 つある。どちらも判定を厳しくする方向にしか働かず、既存の判定を緩めてはいない。
   - アンカーの被覆率 0.20 のゲート
   - 格子の外の 50 µm の漸減（`TAPER_UM`）。この帯の勾配は最大 0.08 で、`taper_gradient_max` として記録している。
   - ただし、どちらの値も根拠となるデータで決めたものではない。
7. 画像の ROI の上限は 2400×2400 px。超えるときは、出力解像度を下げるように促す。それ以上の大きさのタイル分割は未実装。
8. レビューの保存はファイルで行う。研究ストアに統合するには承認が必要。

## 8. 次に研究すべき技術的な問題
1. **隣接切片での変形のアンカー**：8_6 のような隣接切片では、核の一対一対応が成り立たない。そのため、核をアンカーにする方式（RBF）は使えない。組織の境界やランドマークなど、核以外のアンカーが必要。
2. **人の修正が役に立つかの評価**：
   - 手動の修正で TRE が改善するかを、独立のランドマーク（アンカーに使わないもの）で評価する手順が要る。
   - 修正に使った点を評価にも使うと、循環した評価になる。
3. **点の改善ゲートと手動の対応の関係**：
   - 既存の `common_audit` は、最近傍の距離が改善することを要求する。
   - 人が正しく修正しても、最近傍の距離が改善しないと棄却されることがありうる。
   - ゲートを緩めずに扱う方法（例：独立のランドマークを使ったゲート）を検討する。
4. **変形が小さいときの過剰適合**：11-13 で、変形がほとんどないときは、B2＋RBF が Affine より悪くなった。手動修正後の RBF も同じ性質を持つはず。正則化の強さを、データから選ぶ方法が要る。過去の Density / Joint Flow のペナルティは使い回さない。
5. **衝突の解き方**：B1 は一対一を強制しない。一対一の制約を後から加えたとき、精度にどう影響するかは未検証。

## 追記（2026-11-18）：画像の表示モードを Viewer と共有
- 問題：細胞対応レビューのビューアーは `he_mode="nuclear"`, `if_mode="nuclear"` に固定されていた。そのため、Viewer（既定は RGB）からタブを切り替えると、組織画像の色が変わっていた。
- 修正（`cell_review_ui.py` だけを変更）：
  - レビュー側にも「H&E表示」「蛍光表示」の切り替えを置いた。選択肢と表示名は Viewer と同じで、既定は RGB。
  - 状態の元は Viewer の `regx-hem` / `regx-ifm` にした。Streamlit では、同じキーのウィジェットを 2 つ置けない。そこでレビュー側は自分のキー（`regx-crv-hem` / `regx-crv-ifm`）を持ち、描画のたびに Viewer の値をコピーし、変更されたら `on_change` で Viewer のキーに書き戻す。どちらのタブで変えても、もう一方に反映される。
  - `page.py`、`scene.py`、`viewer.py`、計算処理、画像変形、Workflow C / D は変更していない。
- 透明度と重なり方：
  - 両タブとも、同じ描画テンプレートと `scene.build_scene` を使う。既定は overlay・透明度 0.5 で、IF（fixed）が下、H&E（moving）が上。
  - 透明度のスライダーの値は、ビューアーごとにブラウザに保存されるので、タブの間では共有しない。
  - Viewer で fixed を H&E にした場合だけ、上下が入れ替わる。レビューは常に IF が fixed。
  - 表示解像度：レビューは 1.6 µm/px に固定。Viewer の既定値も 1.6 で、変更できる。
- 4 章の変形画像と出力 PNG は、これまでどおり RGB のまま（変形結果のデータなので、表示モードには従わない）。
- テスト：`test_display_modes_shared_and_colours_identical`（slow、AppTest）。
  - 両タブを 1 回の実行で描画し、両ビューアーに渡された画像レイヤーを比べる。JPEG のバイト列と配置行列が一致すること、既定が RGB であること、どちらで変えても両方に反映されることを確かめる。
