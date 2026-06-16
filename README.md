# Dự Án "NGHIÊN CỨU VÀ TRIỂN KHAI MÔ HÌNH HỌC SÂU TRONG BÀI TOÁN NHẬN DẠNG CẢM XÚC ĐA PHƯƠNG THỨC", Tiền Đề Cho Mục Đích Sàng lọc Trầm Cảm Ở Giới Trẻ

Kho lưu trữ này chứa toàn bộ mã nguồn, tài liệu và các thành phần cốt lõi của Đồ án tốt nghiệp đại học, được phát triển bởi nhóm **02 sinh viên** thuộc Khoa Điện tử Viễn thông, Trường Đại học Bách khoa - Đại học Đà Nẵng (DUT).

*   **Thành viên nhóm:** Nguyễn Tùng Chi, Lê Huỳnh Khánh Trâm
*   **Giảng viên hướng dẫn:** TS. Trần Thị Minh Hạnh

Dự án này được chia làm hai giai đoạn phát triển chính và được lưu trữ độc lập tại hai nhánh (branches) khác nhau của kho chứa này để người xem tiện theo dõi cấu trúc:

---

## Nhánh `main` - Phân Hệ Trích Xuất Đặc Trưng Giọng Nói (Giai đoạn 1 - môn học PBL5)

Mã nguồn hiện tại bạn đang xem ở nhánh `main` là phiên bản nền tảng tập trung vào bài toán **nhận diện cảm xúc từ tín hiệu giọng nói âm thanh**.
*   **Chức năng chính:** Trích xuất các đặc trưng âm học cơ bản và xây dựng kiến trúc mạng nơ-ron xử lý tín hiệu âm thanh thô.
*   **Cấu trúc tệp tin:** Gồm các file cốt lõi như `model.py`, `common.py`, `app.py` và file cài đặt môi trường `requirements.txt`.

---

## Nhánh `master` - Hệ Thống Đa Phương Thức Kết Hợp Late Fusion (Giai đoạn Toàn Diện - Đồ án tốt nghiệp)

Để xem toàn bộ mã nguồn hệ thống hoàn chỉnh và tối ưu nhất của đồ án tốt nghiệp, vui lòng **chuyển đổi sang nhánh `master`** trên thanh menu của GitHub.

Nhánh `master` bao gồm 4 mô hình mã nguồn lớn phối hợp đồng bộ:
1.  **`model_voice/`**: Mô hình âm thanh giọng nói nâng cao sử dụng kiến trúc **Dual Network** kết hợp trích xuất đặc trưng **MFCC**.
2.  **`model_speech_to_text/`**: Mô hình phân tích sắc thái ngôn ngữ tự nhiên sử dụng **PhoBERT kết hợp LSTM/Attention** sau khi chuyển đổi giọng nói thành văn bản qua Google STT API và tách từ bằng `pyvi`.
3.  **`model_emotion/`**: Mô hình nhận diện cảm xúc qua biểu cảm khuôn mặt từ dữ liệu hình ảnh bằng cấu trúc mạng mạng **CNN** sâu.
4.  **`code_fusion/`**: Tầng tích hợp thuật toán **Late Fusion** gộp kết quả từ biểu cảm khuôn mặt, văn bản ngữ cảnh và âm thanh giọng nói để đưa ra chẩn đoán cảm xúc cuối cùng với độ chính xác đạt **96%** trên tập dữ liệu 100 video.

---

## Dữ liệu Huấn luyện & Trọng số Mô hình (Model Weights)

Để tuân thủ giới hạn dung lượng lưu trữ của GitHub (dưới 100MB cho mỗi tệp), các tệp trọng số mô hình đã huấn luyện hoàn chỉnh của cả hai nhánh (`model.safetensors`, `best_model.keras`, `best_model_voice_by_vid.pth`...) cùng bộ dữ liệu gốc **không được tải trực tiếp lên đây**.

📧 **Vui lòng liên hệ trực tiếp với tôi để nhận liên kết tài khoản lưu trữ đám mây chứa Trọng số mô hình và Bộ dữ liệu đầy đủ nếu có nhu cầu kiểm thử hệ thống (email: nguyentungchi2311@gmail.com).**
