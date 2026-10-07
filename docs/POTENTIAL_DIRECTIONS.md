# Hướng tiềm năng — chưa phải novelty đã xác lập

Hướng chính là decision-value study trong [NEXT_EXPERIMENTS](NEXT_EXPERIMENTS.md). Không chạy đồng thời tất cả hướng dưới đây.

| Ưu tiên | Câu hỏi và cơ chế để thử | Baseline bắt buộc | Gate mở / điều kiện dừng |
|---|---|---|---|
| P1, sau human pilot | Học policy trình bày toàn bộ/một phần/cảnh báo để giảm decision cost, không chỉ SHAP failure | Fixed warning, release-all, top-1, rank/MC và policy cố định cost-aware | Chỉ mở nếu có outcome người dùng độc lập và dữ liệu đủ; dùng logged randomization/propensity đúng khi học offline. Dừng nếu warning đơn giản ngang |
| P2, diagnostic | Phân biệt lời mô tả về giá trị quan sát với phần suy luận do imputation; reason templates theo evidence provenance | Point explanation và explicit missing mask với độ dài tương đương | Có correctness/rubric ngoài model; không gọi observed contribution là causal effect. Chỉ thử nếu lỗi user study chỉ ra hiểu nhầm observed/inferred |
| P3, robustness | Completion evidence có giữ ích khi missingness/nhóm khách hàng thay đổi? | Donor, conditional imputer, simple mask warning; matched budget | Có dữ liệu và use rights; đánh giá subgroup risk/coverage. Dataset lớn không tự tạo thuật toán mới |

Generic reliability head, stable subsets, feature acquisition và conformal sets đã có prior art. Muốn mở lại phải chỉ ra failure mode mới, objective kinh tế mới được đăng ký trước, đối thủ gần nhất và cơ chế khác cụ thể; không chỉ đổi tên.

One-field acquisition NO-GO trước đây chỉ áp dụng retained-candidate coverage. Nếu tương lai có task decision khác, phải audit headroom mới trên development trước; không dùng điều này làm lý do tự khởi động lại branch đã dừng.

Kết quả nên phân thành VERIFIED_ARTIFACT, REPORTED, PROPOSED và NOT_RUN. Không chuyển giả thuyết thành conclusion. Chưa có claim causal, human-validated benefit hoặc guarantee triển khai.
