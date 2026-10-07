# Kế hoạch tiếp theo — FinancePaper

**PROPOSED / NOT RUN.** Kế hoạch này thay các prompt mở rộng model cũ. Không thay định nghĩa hoặc kết quả lịch sử.

## F0 — Làm ngay trên máy: audit utility và khả năng đánh giá

Giữ nguyên predictor, scores, masks, reasons và policies. Đọc artifact nếu có; nếu thiếu thì ghi MISSING hoặc tái lập trong worktree riêng, không bịa từ summary.

Tạo bảng cho release-all, top-1, MC whole-explanation, rank instability, donor/conditional/both-family Stable-Core. Mỗi dòng lưu dataset/condition/target, số customer/case, valid/failed/released/candidate reasons, coverage >=1/>=2, compute và denominators.

Chạy sensitivity `U=b*valid-c*failed-C_compute-C_delay`, b=1; c/b theo grid đăng ký trước `[0, .5, 1, 2, 5, 10]`. Cost compute/delay chưa biết: báo riêng, không tự gọi milliseconds thành tiền. Vẽ các policy frontier, kể cả baseline thắng. Không chọn budget mới theo kết quả cũ để tạo superiority.

Kiểm tra lại hai ngưỡng hòa vốn mô tả 893/505 và 254/261 từ counts gốc. CI bootstrap theo customer/cluster chỉ khi có dữ liệu cấp đó; aggregate counts không đủ để dựng paired CI thật. Utility chưa phải welfare.

Deliverables dự kiến: `runs/decision_value_pilot/<run_id>/utility_*.csv`, một `RESULTS.md`, `manifest.json`, tests. Chưa có script mới tương ứng trong bản consolidation này.

## F1 — Thiết kế một nhiệm vụ người dùng có đáp án kiểm tra được

Chọn nhiệm vụ chính: **nhận diện một lý do đang trình bày có được thông tin hiện có hỗ trợ đủ để sử dụng hay phải rà soát thêm**. Không yêu cầu ra quyết định cho vay thật.

Tạo bộ vignette có record đầy đủ, record từng phần và quy tắc kiểm tra xác định trước. Expert xây rubric về hỗ trợ bằng chứng và hệ quả quyết định; đáp án không được lấy thẳng từ SHAP revision hoặc score của chính policy đang đánh giá. Nếu dùng vấn đề nhân tạo để có đáp án rõ, ghi synthetic decision task; không gọi là external banking validation. Nếu dùng case benchmark thật, thừa nhận những judgment nào không có ground truth.

4 điều kiện, cùng risk prediction và dữ kiện tài chính:
1. Điểm rủi ro, không explanation.
2. Điểm + point explanation.
3. Điểm + point explanation + cảnh báo missing đơn giản.
4. Điểm + point explanation + uncertainty/completion information theo mapping cố định.

So sánh chính **4 với 3**, không chỉ 4 với 1. Nếu chọn thêm phiên bản partial Stable-Core phải khai báo là secondary arm, không đổi arm sau outcome. Giữ effort/thời gian trình bày tương đương; ghi độ dài thông tin để đánh giá information overload.

Case selection có revised, stable, near-tie và MC false-negative. Không chỉ chọn case có lợi. Cân bằng strata phục vụ power thì lưu inclusion probability; không báo prevalence dân số từ mẫu cân bằng. Group names trung tính: contribution dương của nhóm không tự cho phép nói khách hàng "thu nhập thấp" hoặc "thanh toán kém" khi giá trị không chứng minh điều đó.

Không tiết lộ model identity, MC/rank label, computational revision, outcome sau restore cho reviewer trước lựa chọn. Không hiển thị cả p_before/p_after trong primary decision task vì sẽ tiết lộ future information. Thứ tự/case assignment random, cùng người không nhìn lại cùng case ở arm khác. LLM-generated ratings chỉ dùng kiểm tra UI, tuyệt đối không thay human labels.

## F2 — Preregistration trước thu thập

Primary endpoint: decision accuracy hoặc cost/regret theo rubric độc lập đã khóa; chọn một trước collection. Secondary: thời gian, calibrated self-confidence, request-for-review, agreement trên rubric.

Chọn cỡ mẫu bằng power/precision analysis với effect-size tối thiểu có ý nghĩa được supervisor/partner xác định; không coi 3–5 reviewers là xác nhận population. Có thể làm usability pilot riêng; participants/cases pilot không đi vào confirmatory pool. Preregister hypotheses, exclusions, missing ratings, randomization, stopping và reviewer/case clustering. Có consent/ethics phù hợp trước tuyển người; không tự gửi form hoặc thu thông tin cá nhân trong task coding.

Kiểm định dùng uncertainty theo cả case và reviewer (mô hình mixed-effects hoặc bootstrap hai chiều thích hợp), không xem từng rating độc lập. Không hiệu chỉnh policy bằng chính judgment ở test.

## F3 — Gate

- Tiếp tục nếu uncertainty arm cải thiện quyết định ngoài mức cải thiện của generic missing warning với cost chấp nhận được.
- Nếu chỉ tăng cảm giác tin tưởng, chưa có decision benefit: không claim hữu ích nghiệp vụ.
- Nếu expert không coi grouped changes có ý nghĩa, hạ claim về algorithmic sensitivity; không sửa threshold trên assessment để cứu kết quả.
- Nếu không có người tham gia/phê duyệt: dừng ở protocol và software; ghi HUMAN STUDY NOT RUN. Không chuyển reviewer giả thành nghiên cứu kinh tế.

## Ràng buộc thực thi

Chạy F0 và soạn F1/F2 trước, chưa train model mới. Branch mới, giữ dữ liệu/outputs local. Log tiến độ theo stage, `tqdm`/ETA từ tốc độ thực tế, resume và seed/hash. Cuối run chỉ một báo cáo hiện hành; logs chi tiết nằm trong run directory, không nối dài README.
