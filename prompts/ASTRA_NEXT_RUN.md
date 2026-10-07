# Astra — FinancePaper: decision-value pilot

Đọc AGENTS.md và docs/RESEARCH_STATE.md -> docs/NEXT_EXPERIMENTS.md -> docs/DATA_AND_REPRODUCIBILITY.md. Đây là nhiệm vụ mới thay cho các prompt mở rộng model trước đây. Không train thêm classifier hoặc revision head.

## Làm ngay, không chỉ viết kế hoạch

1. Kiểm tra HEAD, git status và artifacts local. Tạo branch `research/decision-value-pilot` hoặc tên mới không trùng; giữ thay đổi chưa commit.
2. Chạy suite hiện có bằng lockfile tương thích; ghi actual pass/fail và device. Không sao chép test count từ report thành kết quả mới.
3. Triển khai F0 utility audit đọc artifact đã có. Giữ whole-explanation revision và per-reason failure tách biệt. Đối chiếu aggregate với reported counts; không đủ file thì báo rõ và dùng historical worktree để phục hồi/tái lập, không fabricate row-level data.
4. Chạy cost-ratio grid đã định trong NEXT_EXPERIMENTS. Giữ release-all, top-1, mask warning, MC/rank và Stable-Core; report policy frontier và break-even, không gọi utility assumptions là welfare đo được. Bootstrap theo customer/cluster khi có dữ liệu phù hợp.
5. Soạn vignette contract và code export case bundle cho F1: cùng prediction, bốn cách trình bày, nhiệm vụ có đáp án/rubric độc lập. Tách private evaluation labels, full information và policy scores khỏi expert-facing bundle. Test source/target separation, randomization và neutral text. Không gọi một positive group SHAP là một sự kiện kinh tế đã quan sát.
6. Tạo protocol/preregistration draft: primary 4-vs-3, decision-quality outcome, power assumptions, exclusions, clustered analysis, ethics gate. Chưa có participant thì HUMAN STUDY NOT RUN. Không dùng AI ratings thay human data.

## Gate và deliverables

Chỉ hoàn thành F0 + F1/F2 preparation ở lượt này. Không thu thập phản hồi thật, gửi form, chấp nhận terms thay user hoặc chạy mọi hướng tiềm năng.

Lưu code mới, tests, một report trong `runs/decision_value_pilot/<run_id>/`; công bố trong repo chỉ code, protocol và aggregate được phép. Report chỉ rõ VERIFIED/REPORTED/PROPOSED/NOT_RUN; còn thiếu gì, baseline thắng ở đâu, cùng denominator và đúng units.

Progress: stage logs, tqdm/ETA dựa trên đo thực, resume; log chi tiết không vào docs chính. Không cần subagents; nếu dùng phải có công cụ thật và không nhận multiprocessing là peer review.

Kết thúc: push branch, không tự merge main. Tóm tắt tiếng Việt 1) utility audit thực chạy; 2) strong simple baseline; 3) task có outcome độc lập hay chưa; 4) chưa có human evidence; 5) đúng một next action. Không hứa ESWA hoặc method novelty.
