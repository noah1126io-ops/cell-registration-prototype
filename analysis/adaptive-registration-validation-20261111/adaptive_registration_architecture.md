# Adaptive Registration Engine の構成案（研究用、2026-11-11）

図：`figures/fig7_adaptive_architecture.svg`

`Input → Data Profiler → Strategy Planner → Global Search → Refinement → Validation → Artifact`

## 設計の原則（今回の検証結果から）
1. **使える探索手段は、すべて実行して候補を残す。**
   - 根拠：
     - 6_2 の IF を半分にしたケースでは、核探索が誤った位置を UNIQUE（margin 0.88）で選んだ。正解は 2 位だった。
     - 8_6 では逆に、核探索がどの条件でも失敗した。
   - スコアの低さを理由に、片方の手段を黙って捨てない。engine 1.2 の C1 はすでにそうなっている。この性質を維持する。
2. **切片関係のメタデータ、ファイル名からの推測、観測した核の一致を、別々の欄で扱う。**
   - 根拠：6_2 では、メタデータ（DIFFERENT、ユーザーによる訂正）とファイル名からの推測（SAME）が食い違っている。
   - 本番では、ファイル名の推測だけで同一切片のゲートが選ばれ、ACCEPTED になっている。
3. **検証を 5 つの独立した経路に分ける。探索スコアそのものを検証に使わない。**
   - 5 つの経路：組織レベルの一致、核レベルの一致、変換の妥当性、候補の一意性、独立な証拠。
4. **TISSUE_LEVEL を自動で採択しない。**
   - 根拠：鏡像の IF で、TISSUE_LEVEL の誤採択が実際に起きた（`quadrant_validation_audit.md`）。
5. **本番の既定動作を変えない。**
   - 適応型の処理は、研究モードで `plan` を作り、engine 1.2 の既存オプション（`localization`、`section_relation`、`image_feature`）に変換して実行する。

## 各層

### Input
- 必須の入力：
  - 画像のパス、pixel spacing、画像サイズ
  - 両側の核座標
  - 来歴：切片 ID、切片の関係（`SAME` / `DIFFERENT` / `UNKNOWN`）、その記録元
- 現在の `viewer_sample_manifest.json` には `provenance.same_section` がある。しかし、**エンジンには渡されていない**（`infer_section_relation` はファイル名しか見ない）。v1.0 で最初に直すべき点。

### Data Profiler（`data_profiler.py`、仕様は `data_profiler_spec.md`）
- 探索前に測れる指標だけを出す：入力の有無、座標の整合、画像の品質、チャンネルの妥当性、cleft の強さ、外縁の情報、反復性、核の密度、FOV の比、切片関係の根拠。
- 計算量は小さい（16 ケースで数秒。特徴 map は探索と共有）。

### Strategy Planner（`adaptive_strategy.py`、規則は `adaptive_strategy_rules.md`）
- ルールで判断し、出力は JSON の plan（schema は `adaptive_strategy_schema.json`）。
- plan に含めるもの：
  - 実行する手段（使えるものすべて）
  - 主な手段（表示と並べ替えだけに使う）
  - ゲート（`same` / `adjacent` / `both`）
  - ICP の結果を最終変換に使うか
  - 自動採択の可否
  - 危険の flag と、人の確認が必要な理由
- `validate_plan` が、安全上の禁止事項を機械的に検査する：TISSUE_LEVEL の自動採択、`SAME` が確定していない状態での細胞レベルの自動採択、採択していない変換の Workflow D への出力。

### Global Search（engine 1.2 のまま）
- 画像（occ+cleft）と核（E0x）を独立に実行し、上位 6 候補ずつを残す。
- 計算時間（4 CPU）：画像は約 14 分、核は約 7.5 分。これが全体の大部分を占める（図 3）。

### Refinement
- 核の ICP は、`SAME` が確定していて核ゲートを通る場合だけ、最終変換に使う。それ以外は診断用（探索で得た姿勢を維持し、移動量を記録する）。
- 1.2 では `relation=unknown` の場合に ICP の結果を最終変換に使う。案ではこれを診断用に変える（研究モードの plan でのみ）。
- 組織レベルの精密化（非剛体を含む）は v1.1 以降。

### Validation（5 つの経路）

| 経路 | 内容 | 現状 | 循環しないための注意 |
|---|---|---|---|
| 組織レベル | 局所の象限一致。v1.0 案では、判定不能な象限を除外する | 1.2：image#0 のみ、隣接切片のみ | 主探索と同じ特徴を使うので「局所的な安定性」と表示する |
| 核レベル | ≤3 µm の近接率と mirror null（fixed と IF の両方の座標系） | 1.2：同一切片なら合否、隣接切片なら記録のみ | 核探索の候補を核の指標で採択することは、10-19 の規則として事前に定めたもの。画像の候補に対しては独立な証拠 |
| 変換の妥当性 | scale、異方性、反転、重なり率、ICP の移動量 | 一部を記録（flag）。判定はない | — |
| 候補の一意性 | margin、上位候補の分散、小視野・反復の flag | verdict だけ | Profiler の flag と組み合わせて「一意でも信頼が低い」状態を表す |
| 独立な証拠 | 画像の候補と核の候補の一致、ランドマーク、目視 | なし（今回 collect.py で計算） | 一致すれば肯定の証拠。一致しないことは否定の証拠にしない（隣接切片では核が失敗するのが普通） |

### Decision と Artifact
- 判定の状態（`ACCEPTED`、`TISSUE_LEVEL`、`NOT_ACCEPTED`、`AMBIGUOUS`、`CONFLICT`、`ASSISTED_REVIEW_REQUIRED`）は変えない。
- 追加するのは理由の構造化：どの経路が合格し、どれが判定不能だったか。
- Artifact には profile、plan、全候補、全経路の値、判定の理由を保存する。
- Workflow D に渡すのは `accepted=True` の場合だけ（`workflow_d_bridge` は変更しない）。

## 現在のコードで、構造上の問題になる点（Q11 への回答）
1. **切片の関係がファイル名の推測だけで決まる**（`infer_section_relation`）。manifest の来歴は入力に含まれていない。`unknown` は同一切片として扱われる。
2. **1 つの relation の文字列で、C2（ICP を使うか）と C3（どのゲートか）の両方が決まる。** 証拠の経路に分かれていない。
3. **IF のチャンネルの意味が、特徴抽出にハードコードされている**（`features.extract`：B−R を Hoechst とみなし、max(G,R) を非核のシグナルとみなす）。
   - 今の 031 系の表示用 RGB（R=Desmoplakin、G=Dystrophin1、B=Hoechst+Desmoplakin）を前提にしている。
   - **別の染色パネルや別のチャンネル順の画像が来ると、エラーにならずに誤った特徴になる。** 適応型にするうえで最大の障害。
4. **特徴抽出が画像パスと強く結びついている。** 派生条件や特徴のキャッシュを入れる場所がない（今回は `workflow.extract` を差し替えて回避した）。
5. **象限チェックは image#0 にしか実行されず、2 値（合格／不合格）。** 判定不能という状態がない。
6. **計算時間**：1 ペアに約 25 分（4 CPU）。研究用としては許容できる。全候補の監査を加えると、さらに 2〜7 分かかる。GPU は使っていない。
7. `decide` と `_decide_adjacent` が分岐で二重になっている。経路を増やすと保守が難しくなる。v1.0 では「経路ごとの判定 → 方針表で状態を決める」形に整理するのが望ましい。
