# 📜 CORE_RULES.md - Dự án Rabit_Exness AI

## 1. Nền tảng & Kiến trúc (Platform & Architecture)
* **Ngôn ngữ lập trình:** Python 3.10+.
* **Giao tiếp Terminal:** Sử dụng thư viện `MetaTrader5` (hoặc REST API tương đương) để kéo dữ liệu realtime và thực thi lệnh.
* **Cơ chế hoạt động:** Fully-Automated (Tự động 100% từ phân tích, vào lệnh đến quản lý vốn).
* **Mục tiêu Tự tiến hóa (Self-Evolving AI):** Hệ thống khởi chạy bằng Rule-based (5 vũ khí cốt lõi). Song song đó, module Machine Learning (Reinforcement Learning) sẽ chạy ngầm để học hỏi từ lịch sử trade thực tế, tự động tinh chỉnh các trọng số (như hệ số ATR, độ dài nến Pinbar) để tối ưu Winrate theo thời gian.

## 2. Quản trị rủi ro Exness
* **Loại tài khoản:** Exness Standard Cent.

## 3. Khung thời gian & Chiến lược (Multi-Timeframe Logic)
* **H1 & M15 (The Compass):** Khung xác định Xu hướng (Market Structure - BOS/CHoCH) và Vùng chờ giao dịch (Point of Interest - POI) dựa trên SMC Gap/FVG.
* **M5 (The Trigger):** Khung thời gian thực thi lệnh. Bot chỉ kích hoạt quét tín hiệu M5 khi giá các cặp giao dịch forex đã tiến vào vùng POI của M15/H1.

## 4. Định lượng 5 Vũ khí Cốt lõi (The 5 Weapons)
1.  **Market Structure (H1):** Thuật toán nhận diện Swing High/Low chuẩn xác, không vẽ lại (No Repainting).
2.  **SMC Gap / FVG (M15):** Nhận diện Imbalance 3 nến. Box FVG sẽ bị hủy (Mitigated) khi giá lấp đầy.
3.  **Cấu trúc nến Pinbar (M5):** Tỷ lệ râu nến/thân nến phải đạt chuẩn do Mày dùng định nghĩa để xác nhận sự từ chối giá tại POI.
4.  **Volume Spread Analysis - VSA (M5):** Sử dụng **Tick Volume**. Tín hiệu Pinbar chỉ hợp lệ nếu Tick Volume của nến đó cao đột biến (Stopping Volume/Climax) hoặc cực thấp (No Demand/No Supply) so với đường MA20 của Volume.
5.  **ATR (14):** Chỉ báo cốt lõi để tính toán biên độ Stop Loss động (SL = Giá vượt râu nến + Spread + (Hệ số * ATR)) và Take Profit.

## 5. Tiêu chuẩn Code cho AI Coder (Antigravity)
* **Kiến trúc Modular (OOP):** Tách biệt rõ ràng các file logic: `data_pipeline.py` (kéo dữ liệu), `strategy_engine.py` (tính toán 5 vũ khí), `risk_manager.py` (Lỗ tối đa 3% mỗi lệnh), `execution.py` (gửi lệnh) và `ml_model.py` (mô hình AI).
* **Log & Error Handling:** Mọi quyết định vào/ra lệnh, lỗi kết nối hoặc vi phạm rủi ro phải được ghi log chi tiết vào file `system.log` để phục vụ cho quá trình huấn luyện AI sau này.