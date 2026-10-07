# FinancePaper — trạng thái và hướng đã chọn

Cập nhật định hướng: **2026-10-07**. Nguồn kết quả: commit `95773c74f757be9b99afbccb48bcd9d17fd305d7`. Đây là tổng hợp, **không phải một lượt training hoặc human study mới**.

## 1. Quyết định

**Giữ benchmark; chuyển câu hỏi chính sang giá trị thông tin của explanation đối với quyết định.** Chưa chọn một thuật toán XAI mới. Không tiếp tục neural-head/Stable-Core/conformal sweep chỉ để tìm con số đẹp.

Câu hỏi: Khi hồ sơ chưa đầy đủ, cung cấp thông tin về độ bất định của lý do có giúp người sử dụng phát hiện thông tin chưa được hỗ trợ và quyết định rà soát tốt hơn không, so với chỉ có điểm rủi ro hoặc một cảnh báo missing đơn giản?

Một explanation thay đổi có thể là cập nhật hợp lý trước thông tin mới. Không đặt mục tiêu làm mọi explanation bất biến. Điểm dự báo ổn định cũng không chứng minh dự báo đúng.

## 2. Kết quả được giữ

Nguồn chi tiết bất biến: [FINAL_RESEARCH_RESULTS](../archive/20261007/docs/FINAL_RESEARCH_RESULTS.md). Không gộp các mẫu số/target khác nhau.

| MCAR30, grouped top-2 | Taiwan | Polish |
|---|---:|---:|
| Revision trong ca ổn định và đủ điều kiện, raw abs(delta p) <= .02 | 42/727 = 5.78% | 85/552 = 15.40% |
| CI95% | 4.14–7.51% | 12.30–18.41% |
| AP phát hiện revision, prediction-only | .1447 | .2622 |
| AP, rank instability | .6954 | .8219 |
| AP, MC K8 | .6926 | .8580 |

AP trên đây là phát hiện **revision**, không phải default. Sau dung sai khoảng cách rank .02 raw logit, tỷ lệ Region B grouped top-2 MCAR30 còn 2.89%/10.87%. Grouping/tolerance làm giảm hiện tượng nhưng chưa xóa nó; ý nghĩa nghiệp vụ chưa được người thật xác nhận.

Ở một target khác — **meaningful-positive failure theo từng lý do**, hỗn hợp môi trường đã xét:

| Policy | Taiwan: failed/released | Polish: failed/released |
|---|---:|---:|
| Release-all | 564/21,994 = 2.56% | 277/7,100 = 3.90% |
| Both-family Stable-Core | 59/20,596 = .29% | 16/6,585 = .24% |

Stable-Core giảm failure và có ích ở một số điểm đánh đổi, nhưng release-all vốn đã dưới budget 10% và giữ nhiều lý do hơn. Không lấy failure từng lý do thay cho revision cả explanation. Baseline matched-size cho thấy một phần lợi ích về chọn danh tính reason, không thiết lập thuật toán mới.

## 3. Định nghĩa phải giữ nguyên khi tái lập

- Predictor chính: XGBoost 25-view; 25 view là augmentation train, không phải 25 khách hàng độc lập. LR/additive là negative control; GRU là comparator, không phải model thắng.
- TreeSHAP interventional, background cố định, raw logit. Tổng signed SHAP trong group chỉ dùng **cùng những thành viên ban đầu đã quan sát** ở hai đầu. Đây không phải tính lại coalition SHAP.
- Historical revision: một reason top-k ban đầu dương mất dấu/độ lớn hoặc rời top-k theo dung sai đã khóa.
- Meaningful reason failure: group ứng viên hiện tại dương, nhưng attribution sau restore <= .01 raw logit. .01 không phải 1% probability hay ngưỡng kinh tế được xác nhận.
- Natural missing không có truth để restore. Chỉ khôi phục originally observed cells bị che nhân tạo.
- Zero-release có risk không xác định, không phải zero error. Calibration budget không phải guarantee triển khai.
- Taiwan: next-month credit-card default. Polish `5year.arff`: corporate bankruptcy trong một năm, không phải consumer-default replication.

## 4. Những nhánh đã dừng và giới hạn kết luận

| Nhánh | Trạng thái | Không được suy quá phạm vi |
|---|---|---|
| Learned revision selector | Đã thử; thua MC trên setting đã báo cáo | Không chứng minh mọi learned model đều thua; grouped rows còn có selector train cho feature-target rồi recalibrate |
| Stable-Core mới | Chưa có superiority tại budget đã đặt | Không đồng nghĩa không có ích khi chi phí sai reason cao |
| Xác minh thêm một trường | Measured NO-GO trong headroom setting cũ | Không bác bỏ mọi acquisition problem |
| Conformal envelope | Novelty-audit NO-GO cho specification cũ; NOT RUN | Không được gọi là thất bại thực nghiệm |
| TabM/Freddie | NOT RUN | Không có kết quả large-scale hoặc architecture-generalization mới |

Taiwan/Polish đã được xem trong quá trình phát triển. Reuse phải gọi là exploratory. Polish không có company ID đáng tin; exact duplicates chưa bảo đảm độc lập doanh nghiệp. Calibration statement-level không phải entity-level certificate. MC có thể tự tin sai khi donors không phủ giá trị ẩn quan trọng.

## 5. Tái phân tích ngày 07/10: động lực kinh tế, không phải kết quả human study

Bản review trong chat cho ví dụ Shapley chính xác `f(a,b,h,c)=a(1-h)+bh+c`: với a=b=1, c=.5, thay h từ 0 sang 1 giữ probability .817574 nhưng đổi observed top-2 A,C thành B,C. Đây là counterexample toán học, không phải hồ sơ thật hoặc novelty.

Giả định minh họa `U=b*(valid reasons)-c*(failed reasons)`: both-family Stable-Core tránh 505 failures nhưng bỏ 893 valid reasons ở Taiwan; tương ứng 261 và 254 ở Polish. Hòa vốn trước compute/delay khi `c/b > 1.7683` hoặc `.9732`. Các chi phí này **chưa được đo**, utility tuyến tính chỉ là sensitivity analysis. Không dùng chúng để chứng minh thiệt hại tài chính thật.

## 6. Hướng được chọn để chạy tiếp

`Predictor cố định -> point explanation/MC evidence -> các cách trình bày được khóa -> cùng một nhiệm vụ quyết định -> đo chất lượng quyết định và chi phí`.

Bước đầu là audit utility + dựng nhiệm vụ đánh giá có outcome độc lập, không tuyển người hoặc tạo dữ liệu human giả. Sau đó mới preregister và chạy thí nghiệm người dùng theo phê duyệt phù hợp. Xem [NEXT_EXPERIMENTS](NEXT_EXPERIMENTS.md).

**Định vị:** evaluation/human-AI decision support có bằng chứng nền; method mới chưa xác lập. Không hứa ESWA acceptance.
