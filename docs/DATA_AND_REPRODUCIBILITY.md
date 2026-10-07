# Dữ liệu, code và cách chạy

## Nguồn đang có trong nghiên cứu

| Dữ liệu | Vai trò và target | Giới hạn |
|---|---|---|
| UCI Default of Credit Card Clients, 30,000 rows/23 predictors | Consumer credit-card default tháng tiếp theo; bộ chính | Lịch sử đã được nghiên cứu nhiều lần, không phải untouched holdout |
| UCI Polish Companies Bankruptcy, chỉ `5year.arff`, 5,910 statements | Bankruptcy doanh nghiệp trong một năm | Natural NaN không restore; ID doanh nghiệp thiếu; exact-feature clusters chỉ giảm trùng |
| South German Credit, 1,000 rows | Good/bad credit, dữ liệu phụ đã xem xét | Không gộp target với next-month default; không tự suy có experiment đã chạy |
| Freddie SFLLD | Kế hoạch 90+ DPD theo horizon/termination đã mô tả | NOT RUN; phải có official files, receipt, terms và temporal/censoring audit |

Các mô tả trên theo [historical provenance audit](../archive/20261007/docs/DATASET_PROVENANCE_AUDIT.md) và [data policy gốc](../archive/20261007/data/README.md). Đây không phải cấp phép mới. UCI licenses/citations và quyền Freddie phải kiểm tra lại tại thời điểm dùng; không dùng mirror để vượt restriction. Không commit raw personal/financial rows, human responses, credentials, model/cache hoặc outputs nếu chưa rà quyền.

## Code được giữ

`src/`, `scripts/`, `tests/`, `configs/`, `pyproject.toml`, `uv.lock`, `.python-version` không thay đổi trong consolidation. Các test counts trong report là receipt lịch sử, không phải lượt test mới.

Từ repo root, dùng môi trường tương thích lockfile:

```bash
uv sync --frozen --extra temporal
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --extra temporal pytest -q
```

Đọc `--help` của script trước khi chạy. Windows/stages có thể cần artifacts trong `outputs/` vốn không được commit. Không coi thiếu artifacts là run thành công.

## Tái lập lịch sử không ghi đè hướng mới

Lệnh và thứ tự stages đầy đủ nằm ở [README lịch sử](https://github.com/QuocKhanhLuong/FinancePaper/blob/95773c74f757be9b99afbccb48bcd9d17fd305d7/README.md). Một số report writers cũ ghi vào `docs/`; chạy replay trong worktree tại commit lịch sử để không tái tạo tài liệu cũ vào thư mục hiện hành:

```bash
git fetch origin
git worktree add --detach ../FinancePaper-history-95773c7 95773c74f757be9b99afbccb48bcd9d17fd305d7
```

Không tự chạy lại mọi stage để có thêm test counts. Ưu tiên phục hồi verified local artifacts; giữ hash và outcome flags.

## Contract cho run tiếp theo

Mỗi run: `config`, input/source hashes, seeds, dependency/device info, timing, counts, convergence/warnings, current-only và verification-only outputs tách nhau. Single `RESULTS.md` chứa bảng chính, CI, negative results và NOT RUN; terminal logs/worker logs giữ local trong run directory.

Mỗi customer/cluster giữ mọi masks trong cùng partition và bootstrap unit. Không xem repeated episodes là nhiều người mới. Mọi thresholds/loss/budget phải được chọn trước assessment. Human data cần de-identification, consent/ethics và access control riêng.
