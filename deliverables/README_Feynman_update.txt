Feynman original-PySR-score ablation + persistence/replay update

Key controls in run_demo_feynman_dimless.mlx:
  RunPhDNMainModel
  RunOriginalPySRScoreBaseline
  RunMLPBaseline / RunEQLBaseline / RunKANBaseline / RunSINDyBaseline
  displayRecordReport_PhDN
  displayRecordReport_Stage0SR
  displayRecordReport_PySROriginalScore
  displayRecordReport_MLP / EQL / KAN / SINDy
  SaveResults

Results file:
  outputs/Feynman_Dimless/feynman_dimless_results.mat

Run=true takes priority. If Run=false and matching displayRecordReport=true, the saved current (round,case,method) record is loaded, printed, and included in summaries. Each completed method is checkpointed immediately; the final aggregate merge does not erase other method records.

The original-PySR ablation uses selectionPolicy=pysr_native_score at BOTH the Python per-run selector and MATLAB cross-restart merger. It bypasses the proposed machine-floor simplicity and structure/validation composite selection. Search grammar/budget/seeds/data/SINDy bypass remain matched.
