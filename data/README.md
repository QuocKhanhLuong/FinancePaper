# Data provenance

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
