# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 520; đối tượng sau duyệt: 508; ngưỡng accepted: IoU >= 0.9
- Đã **loại 24 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 495 | 495/520 = 95.2% | 520 box máy |
| edited (sửa) | 2 | 2/520 = 0.4% | 520 box máy |
| deleted (xoá) | 22 | 22/520 = 4.2% | 520 box máy |
| added (vẽ thêm) | 11 | 11/508 = 2.2% | 508 đối tượng sau duyệt |

Trong số deleted + added, **7 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 4.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 268 | 1 | 0 | 2 |
| bicycle | 5 | 0 | 2 | 1 |
| motorbike | 137 | 0 | 0 | 2 |
| car | 55 | 0 | 0 | 1 |
| bus | 3 | 0 | 0 | 2 |
| truck | 5 | 0 | 1 | 3 |
| three_wheeler | 22 | 1 | 19 | 0 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 45 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 40 | 31 | 5 | 56.3% (KTC 95%: 45–67%) | 88.9% (KTC 95%: 77–95%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 3/12 = 25.0% (KTC 95%: 9–53%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 1/3
  - wrong_class: 1/3
  - loose: 0/3
  - missing: 1/3
  - lọt: img_005.jpg (loose), img_017.jpg (wrong_class), img_004.jpg (extra), img_035.jpg (loose), img_026.jpg (loose), img_028.jpg (missing), img_009.jpg (missing), img_008.jpg (wrong_class), img_027.jpg (extra)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
