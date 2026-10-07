# Nguồn và phạm vi chứng cứ

## Nguồn nội bộ

- [Báo cáo kết quả đầy đủ](../archive/20261007/docs/FINAL_RESEARCH_RESULTS.md), nguyên văn từ `95773c7`; đây là nguồn số chính.
- [Metric audit](../archive/20261007/docs/REVISION_METRIC_AUDIT.md), [semantic groups](../archive/20261007/docs/DOMAIN_REASON_GROUPS.md), [Stable-Core results](../archive/20261007/docs/STABLE_CORE_RESULTS.md).
- [Conformal novelty audit](../archive/20261007/docs/CONFORMAL_EXPLANATION_NOVELTY_AUDIT.md), [SOTA review](../archive/20261007/docs/SOTA_COMPARISON_2026.md). Giữ kết luận NO-GO có phạm vi, không biến search absence thành first-ever claim.
- Review chat `Finance_Quant_Research_Review_20261007.zip`, file `project_review_20261007/REVIEW_VI.md`: nguồn phép tính hòa vốn/counterexample mới. Các số đã được đưa vào RESEARCH_STATE với nhãn hậu nghiệm; chưa rerun trong consolidation.

## Primary references cần đọc cho hướng mới

- Poursabzi-Sangdeh et al., *Manipulating and Measuring Model Interpretability*, CHI 2021 / arXiv v5: https://arxiv.org/abs/1802.07810 ; DOI 10.1145/3411764.3445315. Thiết kế thí nghiệm cho thấy interpretability và decision quality phải được đo riêng. Trang arXiv đã kiểm tra lại ngày 2026-10-07; không phải một audit toàn literature mới.
- Paes, Wei & Calmon, *Selective Explanations*, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html . Nguồn từ audit lịch sử: selective quality/refinement không phải novelty của repo.
- Vo et al., *Explainability of Machine Learning Models under Missing Data*: https://arxiv.org/abs/2407.00411v3 . Nguồn audit lịch sử về ảnh hưởng imputation lên attribution.
- Golchian & Wright, *Imputation Uncertainty in Interpretable Machine Learning Methods*: https://arxiv.org/abs/2512.17689 . Nguồn audit lịch sử; MI uncertainty không phải thuật toán mới ở đây.

Human decision-value study được chọn là hướng triển khai tiếp, chưa phải gap đã chứng minh không ai làm. Trước preregistration, rà literature cụ thể cho nhiệm vụ và intervention; giữ rõ abstract-only/full-text, năm/version và phạm vi kết luận.
