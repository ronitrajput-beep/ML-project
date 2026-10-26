# Protocol fixed before model results

Question: Can a small, training-only group-outage augmentation improve minority-fault F1 under persistent channel loss, and does the answer survive separating training from threshold selection?

This is an empirical methods study, not a claim to invent missing-data learning. Missingness shift, mask augmentation, boosting and validation-based thresholds are prior ideas. Candidate contribution: an auditable two-dataset comparison separating training augmentation from decision-threshold selection and testing transfer to omitted outage groups.

Data: APS Failure at Scania Trucks (primary, real operational classification; official 60000/16000 split), AI4I 2020 (secondary, synthetic 10000-row condition-monitoring table). These are diagnosis/condition-monitoring tasks, not forecasts of remaining useful life. No claim of physical sensor causality follows from anonymized APS groups. No resampling positives from the test set.

Three principal models: balanced logistic regression, native-missingness histogram gradient boosting (HGB), group-outage-augmented HGB. Cell-masked HGB is an additional training control. HGB settings fixed: 160 boosting rounds, 15 leaves, learning rate .08, minimum leaf 30, L2 regularization 1.0, no early stopping. No held-out test-driven hyperparameter tuning.

Data partitions: APS official test untouched; stratified 80/20 training/validation split of official training. AI4I stratified 60/20/20 partitions. Seeds 11, 29, 47. AI4I identifiers and five failure-type columns excluded. Type is one-hot encoded. All imputers, scalers, and vulnerability ranks fitted on training rows only.

Group ranking: abs difference in class-conditional means divided by pooled within-training standard deviation for each input column; each group gets its maximum column score. APS groups are feature-name prefixes before underscore (a proxy for one measurement/histogram variable); AI4I groups are its five numerical channels. Top three groups define development environments. Ranks four and five are held out of deliberate masking and threshold selection, then tested for transfer. Rankings recomputed per split from training only. This is feature-outage stress testing, not evidence that physical outages follow this ranking.

Training: baseline receives source training rows. Group augmentation replicates training four times: clean and one copy for each top-three group, replacing each selected group with NaN. Copy weights sum to one per original row. Cell control uses three independent masks with the same respective number of numeric cells removed as each group copy (randomly distributed across rows and columns). Each augmented copy keeps the original labels; no new independent observations are claimed. Balanced loss uses original class counts.

Thresholds: clean-only F1 maximization; or equal-environment mean F1 maximization over clean plus top-three group-outage validation sets. Fixed grid .005 through .995, step .005, ties favour the higher threshold. For every HGB training variant report both thresholds to isolate their effects. Logistic uses the clean-selected threshold. All models also report threshold .5 as a descriptive control.

Test environments: clean, each top-five whole-group outages, MCAR .10 and .30. Same test rows and masks across models and thresholds. Primary endpoint: mean positive-class F1 over clean plus three development-family outages. Secondary: clean F1, two deliberately untrained group-outage F1s, MCAR transfer, accuracy/precision/recall, macro/weighted F1, AUPRC, AUROC, confusion matrix, false alarms and APS cost 10FP+500FN (dataset-defined, descriptive).

Statistics: mean and SD over three train/validation splits. APS uses a shared official test, so split runs are not independent test replications. Paired stratified row bootstrap (1000 replicates, fixed seed) of seed-averaged primary performance differences gives conditional test-sampling uncertainty; does not capture between-fleet/dataset uncertainty. AI4I test partitions differ; bootstrap the first predesignated seed only, with mean/SD across all splits separately. No significance fishing. Report all models and conditions, including negative results. No post-test revision of hyperparameters or method within this confirmatory run.

Compute: CPU-only, bounded to four numerical-library threads, no GPU. Record fit/predict wall times, serialized size, versions, host resources, random seeds and prediction arrays. Save per-split model artifacts and thresholds. Retain the protocol hash in every run manifest.

Literature access: scholarly hosts blocked; GitHub author/publisher repositories accessible. Record source type precisely. Do not label a README as a fully read paper or infer worldwide novelty from incomplete coverage. Seek publisher-hosted full text mirrored in publisher GitHub repos. Cite unrefereed repositories separately from peer-reviewed papers.
