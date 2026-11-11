# UOT adoption decision

## Decision

**B. UOT PROMISING, MORE VALIDATION NEEDED**

対象は **UOT conservative** である。UOT primaryをそのまま次期methodとして採用する結論ではない。

## Basis

- source of truthのUOT primary/conservativeは、real datasetで主要値を誤差 `1e-10` 未満で再現した。
- 利用可能な独立real serial-section datasetは1つだけだった。このdatasetでは両UOTがJointより低いNN proxyを示し、全手法が共通safetyをPASSした。
- 50 synthetic scenariosでUOT conservativeはPASS 47/50（94%）、Jointは41/50（82%）。tissue-domain field EPE中央値は2.2889 vs 2.9234 µm、独立landmark TREは2.2557 vs 3.0560 µmだった。
- UOT conservativeのfield EPEはJointに46勝3敗1同等。missingness 0–50%の全levelで中央値がJointより低かったが、優位幅はmissingnessと単調には増えなかった。
- UOT conservativeはunmatched点を中央値0.9901 µm動かし、Jointの0.9405 µmよりわずかに大きい。高missingnessで常に優位なoutlier handlingは証明されなかった。
- UOT primaryはPASS 32/50（64%）、attempted foldover 6/50、attempted Jacobian min最悪値 -1.1475。NNがJointより良いのにfield errorが悪いcaseが19/50あり、NN proxyへの自己適合リスクを確認した。
- UOT conservativeにもattempted foldover 1/50とmissingness 10%で2/5 rejectionがあり、安全gateを外す根拠はない。rejected outputはAffineへ戻り、適用出力のfoldoverは全methodで0だった。

## Adoption criteria disposition

| Criterion | UOT conservative | Disposition |
|---|---|---|
| Safety PASS rate ≥ Joint | real 100%=100%; synthetic 94%>82% | Met in tested data |
| Multiple real datasets | usable real dataset 1 | **Not met** |
| No large within5/10 degradation | real: +4.0719/+3.3285 pp vs Joint | Met on one dataset only |
| No large mutual-NN degradation | real: +2.0089 pp vs Joint | Met on one dataset only |
| Synthetic field error ≥ Joint accuracy | EPE 2.2889<2.9234 µm | Met |
| Stable under missing/unmatched | lower median EPE at all missingness levels; one 60% PASS stratum | Partially met |
| Independent evaluation not worse | synthetic independent TRE improved; no real landmark truth | Partially met |
| No foldover | applied 0/50; attempted 1/50 safely rejected | Met only with safety gate |
| Pathologically reasonable deformation | real p95/max 2.3929/3.8545 µm; review pending | Not independently confirmed |

## Recommended architecture now

**Joint維持**。

Workflow C、Streamlit UI、Workflow Dは変更しない。次の検証ではUOT conservativeを唯一の固定UOT候補とし、少なくとも追加2 real serial-section pairsとblind manual reviewを実施する。UOT→optional Joint cascadeは今回直接評価していないため、現時点の推奨構成にはしない。

