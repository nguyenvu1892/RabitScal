# 🤖 STANDARD PROMPT TEMPLATE FOR AI CODER (ANTIGRAVITY)

Tôi muốn [NHIỆM VỤ CỤ THỂ CẦN LÀM] để mà [MỤC TIÊU/TIÊU CHÍ THÀNH CÔNG CỦA NHIỆM VỤ NÀY].

**I. QUY TRÌNH BẮT BUỘC (VỆ SINH GIT & LOGGING)**
[... Giữ nguyên như cũ ...]

**I.B — LUẬT THÉP: BẮT BUỘC MERGE TRƯỚC KHI SANG TASK MỚI**
> Được ban hành bởi TechLead ngày 2026-03-05 sau sự cố Task 1.1.

Sau khi bất kỳ nhánh (branch) nào được TechLead nghiệm thu và không có lỗi logic, Antigravity BẮT BUỘC phải thực hiện đầy đủ 3 lệnh sau TRƯỚC KHI checkout sang nhánh mới:
```bash
git checkout main
git merge --no-ff feature/<tên-nhánh> -m "[Phase X][Task X.X] MERGE <tên-nhánh> -> main: <mô tả>"
git push origin main
```
Chỉ sau khi `git push origin main` thành công và được xác nhận, Antigravity mới được phép tạo nhánh mới cho Task tiếp theo.

**II. QUY TRÌNH ÉP BUỘC: ANALYZE & REPORT FIRST (SOP)**
[... Giữ nguyên như cũ ...]

**III. YÊU CẦU CHUYÊN MÔN TỪ SẾP VÀ TECHLEAD**
[... Giữ nguyên như cũ ...]

**IV. QUY CHUẨN BÁO CÁO HOẠT ĐỘNG (BẮT BUỘC TRONG MỌI PHẢN HỒI)**
Bất cứ khi nào ngươi hoàn thành một thay đổi trên dự án, hoặc thực hiện một hoạt động code/fix bug nào, ngươi BẮT BUỘC phải trình bày báo cáo tóm tắt nhưng phải đầy đủ đầy đủ để Mày đọc hiểu được vấn đề và ghi vào 1 file walkthrough.md (lưu ý không được xoá hết nội dung cũ) trong thư mục docs/ theo đúng định dạng chính xác sau đây:

**Task [Số thứ tự Task hiện tại. VD: Task 1.2]: [Tiêu đề Task hiện tại]**
* **Nội dung thay đổi/hoạt động:** [Mô tả chi tiết xác đáng những file nào đã sửa, dòng code nào đã thêm]
* **Lý do:** [Giải thích tại sao lại viết logic code như vậy dựa trên kiến thức Trading/SMC/VSA]
* **Đề xuất cải tiến:** [Phân tích góc nhìn của AI để chỉ ra những lỗ hổng tiềm ẩn hoặc ý tưởng nâng cấp module này cho xịn hơn trong tương lai]

**V. QUY TRÌNH TỰ ĐỘNG HÓA (SELF-EVOLVING AI)**
Ngươi phải thiết kế hệ thống sao cho module Machine Learning (ML) có thể tự động học hỏi và tối ưu hóa các tham số của 5 vũ khí cốt lõi dựa trên dữ liệu lịch sử trade thực tế (Backtest & Live Trade).

**1. Thu thập dữ liệu (Data Collection):**
* Ghi lại chi tiết mọi lệnh giao dịch (vào/ra, SL/TP, PnL, Volume, Spread) vào file `data/trade_log.csv`.
* Lưu trữ các thông số cấu hình (Configuration) hiện tại của 5 vũ khí vào `config/current_settings.json` trước mỗi phiên giao dịch.

**2. Huấn luyện mô hình (Model Training):**
* Định kỳ (ví dụ: sau mỗi 100 lệnh hoặc cuối tuần), chạy script `ml_trainer.py`.
* Sử dụng thuật toán Reinforcement Learning để tìm ra bộ trọng số (weights) mới cho các tham số (ví dụ: hệ số ATR, ngưỡng Pinbar, độ dài nến Market Structure) nhằm tối đa hóa lợi nhuận và giảm Drawdown.

**3. Cập nhật chiến lược (Strategy Update):**
* Sau khi huấn luyện, nếu mô hình tìm thấy bộ tham số tốt hơn, tự động ghi đè lên file `config/current_settings.json`.
* Ghi log vào `History.txt` với format: "[[ML_OPTIMIZATION]] Tự động cập nhật tham số [Tên tham số] từ [Giá trị cũ] lên [Giá trị mới] - Lý do: [Kết quả backtest cho thấy hiệu quả hơn]".

**4. Tự động triển khai (Auto-Deployment):**
* Hệ thống phải tự động tải cấu hình mới này vào module Strategy Engine khi khởi động mà không cần sự can thiệp của con Mày.
