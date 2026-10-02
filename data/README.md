# Data provenance

Run `uv run --frozen financepaper download` from the repository root. Data are
downloaded only by this explicit command; loading and tests never fetch data.
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
