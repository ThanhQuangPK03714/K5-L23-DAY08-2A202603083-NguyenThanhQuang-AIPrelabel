# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603083 - Nguyễn Thanh Quang

> Chỉ ghi phần bổ sung so với v1.0. Các quyết định dưới đây dựa trên lỗi và ca khó đã có bằng chứng trong repo; câu hỏi chưa có ảnh phù hợp vẫn được giữ mở.

## 1. Câu hỏi mở đã chốt

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 3 | Hàng nhô ra có thuộc mask xe không? | Mask theo cấu trúc xe và bộ phận gắn cố định; loại hàng rời. Box detection tiếp tục theo quy tắc nhô khoảng 20% ở v1.0 mục 4.1.5. | `img_031.jpg` | CAL-02 |
| 7 | Có đặt confidence tối thiểu để chấp nhận box không sửa không? | Không đặt một ngưỡng confidence để tự accept. Confidence chỉ xếp thứ tự xem; vẫn kiểm tra lớp và độ ôm sát bằng ảnh. Bẫy wrong-class và loose bị bỏ sót cho thấy confidence không thay kiểm tra. | `img_017.jpg`, `img_035.jpg` | CAL-03 |

Các câu hỏi 1, 2, 4, 5, 6 của v1.0 chưa có ca đủ rõ trong bằng chứng hiện tại để chốt; giữ mở thay vì suy đoán.

## 2. Ca khó mới

| Ảnh | Tình huống | Bằng chứng | Quyết định | Quy tắc bổ sung |
|---|---|---|---|---|
| `img_013.jpg` | YOLO26 gọi xe tuk-tuk là `car`; SAM tạo mask từ box sai lớp. | `model_log.csv` r04 và phân tích Part 5 trong báo cáo. | Sửa lớp theo loại xe nhìn thấy trước khi dùng box để tạo mask. | Mục 2: nhận diện xe ba bánh địa phương là `three_wheeler`; không giữ `car`/`truck` chỉ vì model thiếu lớp này. |
| `img_031.jpg` | Hàng rời trên xe đạp chở hàng làm vùng mask khó xác định. | Annotation `bicycle` #447 và ảnh mẫu QC. | Mask theo khung xe/bộ phận gắn cố định; loại hàng rời. | Mục 4.2: không đưa hàng rời vào mask xe. |
| `img_035.jpg` | Bẫy box `motorbike` lỏng đã lọt qua review. | `traps_key.json`, `review_stats.json`; nhãn cuối #496 vẫn dùng box rộng. | So từng cạnh box với phần xe nhìn thấy trước khi accept. | Mục 4.1: sửa box nếu còn nền thừa đáng kể; confidence không thay kiểm tra. |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt xử lý hàng rời trong mask; thêm kiểm tra lớp ba bánh và độ ôm sát box; không dùng confidence làm điều kiện auto-accept. | `img_013` CAL-01, `img_031` CAL-02, `img_035` CAL-03 |
