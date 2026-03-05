# 🤖 STANDARD PROMPT TEMPLATE FOR AI CODER (ANTIGRAVITY)

[cite_start]Tôi muốn [NHIỆM VỤ CỤ THỂ CẦN LÀM] để mà [MỤC TIÊU/TIÊU CHÍ THÀNH CÔNG CỦA NHIỆM VỤ NÀY]. [cite: 1]

**I. [cite_start]QUY TRÌNH BẮT BUỘC (VỆ SINH GIT & LOGGING)** [cite: 2]
Mày phải thực hiện NGHIÊM NGẶT các bước sau trước khi làm bất cứ việc gì:
1. [cite_start]**Chốt sổ Git:** Thực hiện `git add .`, sau đó `git commit -m "Hoàn tất nhánh cũ"`, và BẮT BUỘC phải MERGE toàn bộ thay đổi của branch đang làm việc vào branch `main` (hoặc `master`). [cite: 2]
2. [cite_start]**Khởi tạo nhánh mới:** Sau khi merge xong, tiến hành tạo và chuyển sang nhánh làm việc mới: `git checkout -b [TÊN_BRANCH_MỚI]` [cite: 3]
3. **Ghi log History:** Mở file `History.txt` và ghi nhận sự thay đổi theo format: 
   - [cite_start]"[[TÊN_PHASE]] [Hành động/Thay đổi cụ thể] - Lý do: [Nêu rõ lý do tại sao phải có sự thay đổi này theo góc nhìn tối ưu hoá]". [cite: 3]

**II. [cite_start]QUY TRÌNH ÉP BUỘC: ANALYZE & REPORT FIRST (SOP)** [cite: 4]
[cite_start]Trước khi viết bất kỳ dòng code thực thi nào tác động lên hệ thống, mày BẮT BUỘC phải tuân thủ luồng sau: [cite: 4]
1. [cite_start]Phân tích số liệu, file CSV hoặc file log hiện tại của hệ thống. [cite: 4]
2. [cite_start]Đề xuất các ý tưởng tối ưu (nếu có) dựa trên thuật toán AI và logic Trading. [cite: 5]
3. [cite_start]Viết toàn bộ bài phân tích và Kế hoạch triển khai (Implementation Plan) vào file `docs/walkthrough.md` (hoặc `report.md`). [cite: 6]
4. [cite_start]**DỪNG LẠI TẠI ĐÂY!** Trình bày báo cáo ra khung chat và chờ TechLead/Sếp review file đó. [cite: 7]
[cite_start]Mày KHÔNG ĐƯỢC PHÉP đụng vào code base cho đến khi nhận được lệnh "PROCEED" từ tao. [cite: 8]

**III. [cite_start]YÊU CẦU CHUYÊN MÔN TỪ SẾP VÀ TECHLEAD** [cite: 9]
- [Gạch đầu dòng 1: Yêu cầu logic cốt lõi...]
- [Gạch đầu dòng 2: Ràng buộc kỹ thuật cụ thể...]
- [Gạch đầu dòng 3: Các lưu ý đặc biệt về hiệu năng/xử lý lỗi...]

[cite_start]Đầu tiên, hãy đọc hoàn toàn các tệp này trước khi phản hồi: [cite: 9]
- [cite_start]`[tên_tệp_1.md/py]` — [Nội dung cốt lõi của tệp này là gì] [cite: 9]
- [cite_start]`[tên_tệp_2.md/py]` — [Nội dung cốt lõi của tệp này là gì] [cite: 9]

[cite_start]Đây là tài liệu tham khảo cho những gì tôi muốn đạt được: [cite: 9]
[cite_start][Tải lên tệp tham khảo dưới dạng markdown, hoặc dán mã nguồn/logic tham khảo vào đây] [cite: 9]

[cite_start]Dưới đây là lý do tại sao tài liệu tham khảo này lại hiệu quả: [cite: 9, 10]
- [cite_start]Luôn luôn: [Phân tích điểm mạnh 1 - Tông giọng, cấu trúc, mẫu thiết kế...] [cite: 10, 11]
- [cite_start]Luôn luôn: [Phân tích điểm mạnh 2...] [cite: 10, 11]
- [cite_start]Không bao giờ: [Những sai lầm cần tránh từ tài liệu này...] [cite: 11]

[cite_start]Đây là những gì tôi cần cho phiên bản của mình: [cite: 11]

[cite_start]**TÓM TẮT THÀNH CÔNG** [cite: 11]
* [cite_start]**Loại đầu ra + độ dài:** [Ví dụ: Code Python thuần, hay Báo cáo Markdown, độ dài bao nhiêu?] [cite: 11]
* [cite_start]**Phản ứng của Mày nhận:** [Mày đọc/TechLead sẽ đánh giá cao điều gì nhất?] [cite: 11]
* [cite_start]**KHÔNG giống như:** [Ví dụ: AI chung chung, code rườm rà, thiếu comment, nhiều thuật ngữ...] [cite: 11]
* **Thành công có nghĩa là:** [Mã nguồn chạy không lỗi? Build thành công? [cite_start]Backtest ra kết quả?] [cite: 11, 12]

[cite_start]Tệp ngữ cảnh của tôi chứa các tiêu chuẩn, ràng buộc, những nguy cơ tiềm ẩn (landmines), và đối tượng khán giả của tôi. [cite: 12] [cite_start]Hãy đọc nó hoàn toàn trước khi bắt đầu. [cite: 13] [cite_start]Nếu bạn sắp phá vỡ một trong các quy tắc của tôi, hãy dừng lại và báo cho tôi biết. [cite: 13]

[cite_start]KHÔNG bắt đầu thực hiện ngay lập tức. [cite: 14] [cite_start]Thay vào đó, hãy đặt các câu hỏi làm rõ cho tôi (sử dụng công cụ 'AskUserQuestion') để chúng ta có thể cùng nhau tinh chỉnh cách tiếp cận từng bước. [cite: 14]

[cite_start]Trước khi bạn viết bất cứ điều gì, hãy liệt kê 3 quy tắc quan trọng nhất đối với nhiệm vụ này từ tệp ngữ cảnh của tôi. [cite: 15] [cite_start]Sau đó, hãy cho tôi kế hoạch thực hiện của bạn (tối đa 5 bước). [cite: 16]

[cite_start]Chỉ bắt đầu công việc sau khi chúng ta đã thống nhất. [cite: 17]