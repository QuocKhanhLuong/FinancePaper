# FinancePaper

**Hướng hiện hành: giá trị của thông tin giải thích khi hồ sơ tài chính chưa đầy đủ.** Giữ benchmark và predictor; chưa có thuật toán XAI mới được xác lập.

> Không chỉ hỏi "lý do có thay đổi không?" mà hỏi "việc trình bày sự bất định đó có giúp người sử dụng quyết định tốt hơn không?"

## Đọc nhanh

| Tài liệu | Nội dung |
|---|---|
| [Trạng thái nghiên cứu](docs/RESEARCH_STATE.md) | Ý tưởng, findings, metrics, các nhánh dừng và hướng được chọn |
| [Thí nghiệm tiếp theo](docs/NEXT_EXPERIMENTS.md) | Utility audit chạy local trước; nhiệm vụ quyết định và human-study gates |
| [Hướng tiềm năng](docs/POTENTIAL_DIRECTIONS.md) | Ba hướng có điều kiện; không phải danh sách model để chạy hàng loạt |
| [Dữ liệu và tái lập](docs/DATA_AND_REPRODUCIBILITY.md) | Nguồn, quyền dùng, artifacts, commands và run logging |
| [Nguồn](docs/SOURCES.md) | Bằng chứng gốc và prior art |
| [Prompt Astra](prompts/ASTRA_NEXT_RUN.md) | Nhiệm vụ giới hạn cho lượt chạy tiếp |

## Những gì đã biết

- Grouped top-2 MCAR30, xác suất chỉ đổi <=2 điểm phần trăm: 42/727 ca Taiwan và 85/552 ca Polish vẫn revise. Đây là denominator stable/eligible, không phải toàn bộ khách hàng.
- MC/rank evidence phát hiện revision tốt hơn prediction-only đã thử. MC không thắng rank instability tổng quát.
- Stable-Core giảm failure từng lý do nhưng chưa có ưu thế coverage tại budget cũ; release-all đã đạt budget trong một số setting.
- Generic selector, one-field acquisition và conformal-envelope specification chưa tạo method claim. Conformal là NO-GO qua audit, không phải experiment thất bại.
- Freddie/TabM/human study: **NOT RUN**. Taiwan/Polish đã được xem trong phát triển; không gọi replay là independent confirmation.

Nguồn bảng và phạm vi: [báo cáo lịch sử](archive/20261007/docs/FINAL_RESEARCH_RESULTS.md). Consolidation ngày 2026-10-07 không train thêm model hoặc thay kết quả.

## Chạy tiếp

Đọc prompt và chạy **F0: utility/headroom audit + chuẩn bị nhiệm vụ quyết định**. Không tự tạo human judgments hoặc tuyển người trước consent/ethics. Không thêm architecture trước khi baseline warning đơn giản bộc lộ hạn chế.

```bash
uv sync --frozen --extra temporal
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --extra temporal pytest -q
```

Đây là lệnh kiểm tra code hiện có; không phải receipt đã chạy trong lần cập nhật tài liệu. F0 runner mới chưa được triển khai.

## Lịch sử không bị xóa

README và toàn bộ docs cũ đã chuyển sang [archive](archive/README.md); code, tests, configs và lockfile giữ nguyên. Nhánh `archive/pre-consolidation-20261007` giữ snapshot trước dọn. Các logs/plan cũ không còn là hướng dẫn hiện hành. Reproduction lịch sử nên dùng worktree của commit `95773c7` để tránh report writers cũ ghi đè docs mới.
