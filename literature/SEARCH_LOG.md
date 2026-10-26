# Discovery and access log

Scholar API/direct PDF probes for arXiv, OpenReview and PMLR returned proxy tunnel 403 in this environment. GitHub and its raw-content host were accessible. Original authors' repositories were inspected first, followed by PMLR's own publication repositories. Failed access was never interpreted as no prior work.

Actual GitHub repository searches during this run:

- tabular classification missing values paper
- tabular missingness shift paper
- classification missing features robust paper
- AI4I 2020 predictive maintenance
- APS failure scania dataset
- tabular benchmark paper pdf
- missing values NeurIPS 2024
- missingness shift
- missing features tabular robustness
- missing data classification paper pdf
- robust tabular missing values 2025
- tabular benchmark Grinsztajn
- NeurIPS missing data paper
- NeuMiss
- NeuMISE
- StableMiss
- missing data robustness paper
- tabular missing features dropout paper (HTTP 429)
- predictive maintenance AI4I paper pdf (HTTP 429)

The 429 responses were not retried in a loop. These are GitHub discovery queries, not scholarly database searches, and search-result absence is not used as novelty evidence. Earlier HTML responses remain in the external research workspace; this log retains the exact query strings.

Publisher metadata was then searched locally for missing-data, missing-value, imputation and tabular terms in PMLR volumes 119 (2020), 162 (2022), 206 (2023), 235/238 (2024), and 258/267 (2025). Ten relevant peer-reviewed PDFs were retrieved from the publisher's GitHub mirror. Newer volumes store PDFs under `main/assets/` while metadata is on `gh-pages`; older volumes place PDFs directly on `gh-pages`. Initial wrong-path 404s were diagnosed and corrected rather than interpreted as paywalls.

Primary gap falsification: DAMS already proposes source masking; TabFSBench already benchmarks feature deletion and feature-importance-related degradation; MINTY/MA learning already reduce dependence on missing variables. Broad claims that these ideas are new were rejected. The paper makes a narrow empirical contribution and no claim of exhaustive worldwide novelty verification.
