# Thông tin học viên

- Họ và tên: Nguyễn Thanh Quang
- MSHV: 2A202603083
- Lớp: L2
- Tài khoản GitHub: https://github.com/ThanhQuangPK03714/K5-L23-DAY08-2A202603083-NguyenThanhQuang-AIPrelabel
- Repo bài nộp (URL): (https://github.com/ThanhQuangPK03714/K5-L23-DAY08-2A202603083-NguyenThanhQuang-AIPrelabel)
- Commit chốt bài (hash): sẽ điền sau commit cuối
- Máy: Windows 11; PyTorch 2.15.0+cpu; không có CUDA GPU được PyTorch nhận diện
- CVAT (Docker): 2.81.0 là phiên bản ghim trong hướng dẫn repo; phiên bản thực chạy chưa xác minh

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 495 / 2 / 22 (mẫu số: 520 box máy); added: 11 (mẫu số: 508 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 40/71 = 56.3% / 40/45 = 88.9% (4 ảnh gold; in-sample)
- Bắt bẫy: 3/12 = 25.0% (dưới ngưỡng 80%)
- Thời gian (ước lượng hồi cứu, không phải số đo có timer): gold_manual 35 phút/4 ảnh = 8.8 phút/ảnh; batch_review 90 phút/24 ảnh = 3.75 phút/ảnh
- Lỗi QC: 2/5 ảnh = 40.0%; 2/102 đối tượng = 2.0% trong mẫu QC hiện tại
- **Quyết định:** Hold — còn câu hỏi guideline chưa chốt; QC cũng phát hiện lỗi và bắt bẫy chỉ 3/12, thấp hơn ngưỡng. Không phát hành nhãn này.
