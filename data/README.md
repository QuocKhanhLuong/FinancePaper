# Data provenance

## Publication policy

The [2026-10-03 provenance audit](../docs/DATASET_PROVENANCE_AUDIT.md) separates
legal license permissions from this repository's stricter artifact policy.
The table answers **CAN this repository commit these artifacts?**

| Dataset | Raw data | Processed rows | Aggregate results | Model artifacts |
|---|---|---|---|---|
| Taiwan UCI | NO | NO | YES, with attribution | NO |
| Polish UCI | NO | NO | YES, with attribution | NO |
| Freddie SFLLD R47 | NO | NO | YES, reviewed noncommercial, non-identifying, non-reconstructive outputs only | RESTRICTED; no weights/reference banks committed |
| Rejected large-dataset candidates | NO | NO | NO data-derived experiments authorized | NO |

Taiwan (DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)) and Polish (DOI
[10.24432/C5F600](https://doi.org/10.24432/C5F600)) are CC BY 4.0: their licenses
allow attributed sharing/adaptation, but this Git repository excludes row data,
caches and model artifacts. Preserve original attribution and indicate changes.
Freddie is governed by provider terms, not Creative Commons. Do not upload loan
rows or local model/background artifacts to external AI services or mirrors.

## Freddie: user-acquired files only

Official source: [Freddie SFLLD](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
Only Release 47 Standard annual samples are selected. The researcher must register,
accept applicable terms, manually obtain the exact files, and complete a local
receipt using `configs/freddie_acquisition.example.json`. An example receipt is
deliberately invalid until actual access details are supplied. Do not claim to have
accepted terms on another person's behalf. No download automation is provided.

Run `uv run --frozen python scripts/prepare_freddie.py --help`; follow the
[staged protocol](../docs/FREDDIE_DATA_PROTOCOL.md). ZIPs remain in a private local
directory; outputs go under ignored `data/processed/freddie_r47_*`. These include
private loan identifiers, feature rows, natural masks, targets and detailed intake
manifests. Nothing is automatically staged or uploaded. Checksums/aggregate metadata
can be considered for publication after review; no invented expected hashes.
Do not publish row-level examples, transformed rows, completions, tree weights or
SHAP backgrounds. A format conversion is not anonymization.

The protocol uses originally observed cells for artificial verification; naturally
unknown values remain unknown. Target files contain future outcomes and must be
excluded from preprocessing, completion fitting and serving APIs. See
[target definition](../docs/FREDDIE_TARGET_DEFINITION.md) and
[citation/access statement](../docs/DATASET_CITATION.md).

## Taiwan

Run `uv run --frozen financepaper download` from the repository root for Taiwan.
Loading and tests never fetch data; download entry points are explicit.
The raw workbook is excluded from Git.

- Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- [Official archive](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip).
- Archive SHA256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
- Workbook SHA256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
- Expected official shape: 30,000 rows, 23 predictors, one ID, one target.
- Target: `default payment next month`, where 1 denotes default.

The first XLS row contains short variable labels; the second contains the field
names. The loader uses the second row, removes ID from predictors but retains it
as `record_id`, and preserves source row order. All original data must be finite
and complete. Monetary fields and AGE are numeric; SEX, EDUCATION, MARRIAGE and
all PAY repayment-status fields are categorical. Unusual codes (including 0,
negative repayment codes and EDUCATION 5/6) remain separate categories.

An existing valid workbook is reused without network access. A checksum mismatch
fails rather than silently accepting a different dataset. If downloading
manually, put the official workbook at `data/raw/default of credit card clients.xls`.
An explicit local CSV may also be supplied with the same 25 column names,
including ID and the target; its fingerprint and nonofficial status are recorded.
Synthetic CSV fixtures are for software validation, not empirical evidence.

## Polish corporate-bankruptcy validation

The explicit `scripts/run_decisive_validation.py external-fit` stage downloads
only the official UCI archive and extracts only `5year.arff` into
`data/raw/polish/`. It verifies both archive and member SHA256 before use.

- [Official UCI source](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
  creator Sebastian Tomczak (2016), DOI 10.24432/C5F600, CC BY 4.0.
- Archive SHA256: `17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9`.
- ARFF SHA256: `cb3f6f250ac46bd8d18e9a222f489fe8ee3e396fcec18959f5a0ef8e8169b2fc`.
- 5,910 statements, 64 numerical ratios, 410 bankruptcy events within one year.
- 4,666 natural missing cells in 2,879 records; 60 duplicate-feature rows.
- Naturally missing cells remain unknown. Only originally observed cells may be
  artificially hidden and verified. Exact duplicates stay in the same partition.

This target is corporate bankruptcy, not consumer credit default. See the
[frozen external protocol](../docs/EXTERNAL_DATASET_PROTOCOL.md) and
[separate external results](../docs/EXTERNAL_VALIDATION_RESULTS.md). Do not pool
the five ARFF horizons or commit raw data, caches, models or generated outputs.
