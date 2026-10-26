# Exploratory sensitivity, fixed after confirmatory results and before this sensitivity

The training-only audit found that four of the top-five APS variables were almost perfectly correlated (bb_000, bu_000, bv_000, cq_000). Their separate deletion may leave almost the same information available. This is a post-hoc diagnostic, not a change to the confirmatory protocol.

On each APS training split, compute pairwise Pearson correlations (at least 1000 jointly observed rows) among the top-five ranked single-column groups. Connect groups if absolute correlation exceeds 0.9999. Select the first ranked connected component with at least two members, remove every variable in it jointly from the official test set, and evaluate the already fitted models at their already fixed thresholds. No retraining or threshold selection. Report descriptive split mean/SD only. This is not evidence that those counters share a physical sensor or fail simultaneously.

## Additional clean-repeat control

After identifying the minimum-leaf-count caveat in the manuscript, fit a separate HGB on four exact clean copies per training row with weight 1/4. Use exactly the existing splits, learner settings, group ranks, threshold grids, and test environments; no HPO. This isolates whether merely changing representation counts can change results despite preserving total weight. Record all three threshold policies, both datasets and all three seeds. This control is post-hoc and remains outside the 24-fit confirmatory archive. Do not replace or reselect any original model.
