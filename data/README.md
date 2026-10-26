# Dataset provenance

Raw data are deliberately excluded from Git. Run `python scripts/download_data.py` from the repository root. `manifest.json` pins public mirror commits and SHA-256 hashes. The downloader checks against the committed hashes before accepting an existing or newly downloaded file.

## APS Failure at Scania Trucks

- Original source: Scania CV AB, donated by Tony Lindgren and Jonas Biteus, September 2016.
- UCI source page: https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks
- Actual acquisition: https://github.com/kev-wes/pdm_scania/tree/4cd68dd63886416e8e60f1330d79ac7f5094ce9a
- 60,000 source records (1,000 positives); official 16,000 test records (375 positives); 170 numeric inputs.
- `na` denotes missing values. Positive means failure of the APS component; negative means failure of another component, not a healthy vehicle.
- The original description distributed with the mirror asserts Scania copyright and GNU GPL v3-or-later terms. That description is preserved as `APS_SOURCE_DESCRIPTION.txt`. A mirror's repository license must not be substituted for original data terms. This project does not redistribute the tables.

## AI4I 2020 Predictive Maintenance

- Original source page recorded in the mirror: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
- Actual acquisition: https://github.com/rockyfjord/predictive-maintenance/tree/789f44a7336f173cea4d4879fb61993db0d42fdf
- 10,000 synthetic records, 339 machine failures; five numeric channels plus product type.
- Exclude UDI, Product ID and the TWF/HDF/PWF/OSF/RNF failure-mode columns. Product type is one-hot encoded.
- Original UCI data licence was not independently retrievable in this environment. No raw table is redistributed; the mirror's MIT code licence is not asserted to cover the source dataset. Verify original terms before any separate redistribution.

These are recorded-condition classification tasks. The tables and this evaluation do not demonstrate future-failure lead time or remaining-useful-life prediction.
