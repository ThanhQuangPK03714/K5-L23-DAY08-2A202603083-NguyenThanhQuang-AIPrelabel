# Báo cáo Day 08 — 2A202603083 - Nguyễn Thanh Quang

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm mẫu số và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao |
|---|---|---|---|
| person | YOLO26m | 0.3 (mặc định) | Recall tối đa 89%, chưa đạt mục tiêu 90%. |
| bicycle | YOLO26m | 0.3 (mặc định) | Chỉ có 4 vật thể Gold, mẫu quá ít để hiệu chỉnh riêng. |
| motorbike | YOLO26m | 0.3 (mặc định) | Recall tối đa 78%, không đạt mục tiêu 90%. |
| car | YOLO26m | 0.3 (mặc định) | Recall tối đa 88%, không đạt mục tiêu 90%. |
| bus | YOLO26m | 0.3 (mặc định) | Không có vật thể Gold, không đủ dữ liệu đánh giá. |
| truck | YOLO26m | 0.3 (mặc định) | Không có vật thể Gold, không đủ dữ liệu đánh giá. |
| three_wheeler | YOLOE-26s | 0.15 | TP=6, FP=3, FN=0; Recall 100%, Precision 67% trên 6 vật thể Gold. |

- Recipe trong `model_log.csv`: r01 YOLO26m pilot/conf 0.25; r02 YOLO26m pilot/conf 0.5; r03 YOLOE-s pilot/conf 0.25; r04 YOLO26m + SAM 2.1 B trên `img_004` và `img_013`; r05 YOLO26m all/conf 0.05; r06 YOLOE-s all/conf 0.05. Recipe threshold đề xuất từ sweep là YOLO26m conf mặc định 0.3 và YOLOE-s `three_wheeler=0.15`.
- Lệnh `merge_prelabels.py` đã dùng: không có lệnh trong log và chưa có `work/prelabel/merged_review/pre.coco.json`, nên không thể xác nhận lệnh thực tế. Lệnh tái tạo theo các file hiện có và threshold sweep là:
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- SAM đã chạy: SAM 2.1 B (`sam2.1_b.pt`) trong r04, theo kiểu detector box → segmenter, tạo 16 mask trên 2 ảnh. Đây là lần thử pipeline tạo mask tự động, không phải bài SAM click ở Part 5. Quan sát `viz/`: ở `img_013`, box/lớp `car` bao rộng mái và thân tuk-tuk, mask chồng lên người ngồi; SAM không sửa được lỗi lớp của detector.
- Part 5 chọn SAM 2.1 B (`sam2.1_b.pt`) cho click vì cùng phiên bản đã chạy được trên CPU ở r04 và hỗ trợ click dương/âm. Ba ảnh click đã thử: `img_013`, `img_001`, `img_002`; chưa có phép so sánh với SAM 1 nên chưa kết luận phiên bản này tốt nhất.
- Lỗi không sửa được chỉ bằng đổi ngưỡng: YOLO26 là detector closed-set, không có lớp `three_wheeler`; tuk-tuk/xích lô có thể bị gán `car` (ví dụ `img_013`). Cần model/lớp phù hợp như YOLOE và vẫn phải kiểm tra nhãn.
- Ví dụ điểm số trong `viz/`: YOLOE pilot ở `img_032` gán xe đạp bên phải thành `three_wheeler` với score 0.27; box trung tâm có score 0.92 bao xe ba bánh rõ hơn. Chưa có ví dụ được ghi nhận về false positive điểm cao.
- Giới hạn Gold: `gold_manual.json` có 47 annotation trên 4 ảnh, gồm 45 vật thể thuộc lớp đối tượng và 2 `ignore_region`. Threshold sweep tính trên 45 vật thể; vì chọn và đánh giá trên cùng Gold nên Precision/Recall có thể lạc quan và chưa đại diện cho toàn bộ dữ liệu.

Nguồn: `submission/labels/gold_manual.json`, `submission/qc/threshold_sweep.csv`, `submission/qc/threshold_sweep.md`, `submission/model_log.csv`, `work/prelabel/t_yoloe/viz/img_032.jpg`, `work/prelabel/t_sam/viz/img_013.jpg`.

## 2. AI error profile (từ qc/review_stats.md)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 495 | 495/520 = 95.2% | 520 box máy, đã loại 24 đối tượng nằm quanh bẫy |
| edited | 2 | 2/520 = 0.4% | 520 box máy |
| deleted | 22 | 22/520 = 4.2% | 520 box máy |
| added | 11 | 11/508 = 2.2% | 508 đối tượng sau duyệt |

- `three_wheeler` là lớp có nhiều box bị xóa nhất: 19/22 box deleted; 22 được accepted, 1 edited. Đây là lỗi nhãn/mapping nổi bật, phù hợp với ví dụ tuk-tuk bị gọi `car` ở `img_013` và xe đạp bị YOLOE gán `three_wheeler` ở `img_032`.
- Trong deleted + added có 7 đổi lớp; ước tính số vật máy bỏ sót thật là 11 added − 7 đổi lớp = 4.
- So với Gold: có threshold sweep theo lớp, không có một Precision/Recall tổng. Tại `three_wheeler=0.15`: TP=6, FP=3, FN=0, Precision=0.667, Recall=1.000 trên 6 Gold. Tối đa recall trong ngưỡng ≤0.5: person 0.889 (16/18), motorbike 0.778 (7/9), car 0.875 (7/8); bicycle có 4 Gold nên không hiệu chỉnh riêng; bus/truck không có Gold.
- Số tham chiếu Gold: 47 annotation trên 4 ảnh, gồm 45 vật thể và 2 `ignore_region`.

Nguồn: `submission/qc/review_stats.md`, `submission/qc/review_stats.json`, `submission/labels/batch_reviewed.json`, `submission/labels/gold_manual.json` và `submission/qc/threshold_sweep.csv`.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 3/12 = 25% (ngưỡng yêu cầu ≥80%; KTC 95% 9–53%). Theo loại: extra 1/3, wrong_class 1/3, loose 0/3, missing 1/3.
- 9 bẫy lọt: `img_005` loose, `img_017` wrong_class, `img_004` extra, `img_035` loose, `img_026` loose, `img_028` missing, `img_009` missing, `img_008` wrong_class, `img_027` extra.
- Accepted cao (95.2%) nhưng bắt bẫy thấp (25%) cho thấy nhiều box thật trông đủ hợp lý để được giữ, song người duyệt vẫn bỏ qua lỗi cài có chủ đích; accepted không đo được khả năng phát hiện lỗi ẩn. Cần quét ảnh trước khi nhìn box máy và rà riêng các lỗi loose/missing.

## 4. Máy có đáng dùng không?

- Ước lượng hồi cứu (không có timer log): gán tay Gold 35 phút/4 ảnh = 8.8 phút/ảnh; duyệt batch có pre-label 90 phút/24 ảnh = 3.75 phút/ảnh. Đây là ước lượng sau buổi làm, không phải phép đo thời gian thực.
- Thời gian inference trong `model_log.csv` không thay thế thời gian gán tay/duyệt: YOLO26 pilot 0.74–0.75 giây/ảnh, YOLOE pilot 0.34 giây/ảnh, SAM 2.1 B 5.93 giây/ảnh; đây là số trên CPU.
- Nguồn: `submission/time_log.csv` (ước lượng hồi cứu), `submission/model_log.csv` (thời gian inference).
- Thiên lệch: Tôi đã gán nhãn Gold trước nên có thể đã quen với hình ảnh và đặc điểm đối tượng. Điều này có thể ảnh hưởng đến tính khách quan khi đánh giá pre-label.
- Kết luận: ước lượng hồi cứu gợi ý review có pre-label mất khoảng 3.75 phút/ảnh; không có timer đối chứng cho cùng 24 ảnh gán tay nên chưa thể kết luận AI tiết kiệm bao nhiêu. Threshold sweep gợi ý `three_wheeler=0.15` đạt Recall 100% nhưng Precision 67% trên 6 vật Gold; mẫu nhỏ và có nguy cơ lạc quan do chọn/nghiệm thu trên cùng Gold.

Ba quy tắc giảm thiên kiến:

1. Kiểm tra toàn bộ ảnh thay vì chỉ kiểm tra những box AI đã dự đoán.
2. Không mặc định chấp nhận box chỉ vì confidence cao; phải kiểm tra vị trí, kích thước và class.
3. Ghi lại mọi trường hợp sửa, xóa và thêm box để có số liệu đánh giá AI error profile.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh / đối tượng | SAM (phiên bản) | Số click | Kết quả và giới hạn | File trong masks/ |
|---|---|---:|---|---|
| `img_013` tuk-tuk (xe) | SAM 2.1 B (`sam2.1_b.pt`) | 13 (4 dương, 9 âm) | Mask xe đã loại phần lớn người nhìn thấy; rà lại biên mái và các thanh khung, chưa đo IoU. | `masks/tuktuk_img013.jpg` |
| `img_013` người lái | SAM 2.1 B (`sam2.1_b.pt`) | 6 (3 dương, 3 âm) | Có mask riêng cho người lái; biên tại chỗ tiếp xúc ghế/xe vẫn cần kiểm tra. | `masks/tuktuk_driver_img013.jpg` |
| `img_001` pickup | SAM 2.1 B (`sam2.1_b.pt`) | 6 (4 dương, 2 âm) | Biên xe khá gọn, nền ngoài xe được loại; chưa đo IoU. | `masks/pickup_img001.jpg` |
| `img_002` motorbike (phần xe nhìn thấy) | SAM 2.1 B (`sam2.1_b.pt`) | 6 (2 dương, 4 âm) | Mask xe giới hạn vào phần xe lộ ra; phần bị người che không được đoán. | `masks/motorbike_img002.jpg` |
| `img_002` người lái | SAM 2.1 B (`sam2.1_b.pt`) | 6 (3 dương, 3 âm) | Có mask riêng cho người lái; vùng tay tiếp xúc tay lái cần rà biên thủ công. | `masks/motorbike_rider_img002.jpg` |

- Đối chiếu guideline mục 4.2: người và phương tiện chạm nhau được lưu thành các mask riêng. Tuk-tuk và xe máy bị che khuất nên các mask chỉ mô tả vùng nhìn thấy; ảnh overlay là bằng chứng thao tác, không thay phép đo IoU hoặc bản polygon đã kiểm tra từng đỉnh.
- Interactor SAM trong CVAT: chưa có ghi chép chạy bên trong CVAT; các mask trên được tạo bằng `scripts/sam2_points.py` với SAM 2.1 B trên CPU.

## 6. Calibration (Part 6)

- Đã ghi 3 quyết định ở `qc/calibration_decisions.csv`: lỗi model gán xe ba bánh (CAL-01, `img_013`), câu hỏi mask cho hàng rời (CAL-02, `img_031`), và bẫy box lỏng lọt review (CAL-03, `img_035`).
- Quy tắc tương ứng đã được thêm vào `guidelines/guideline_v1.1_calibrated.md`. Không đặt confidence tối thiểu để auto-accept; vẫn kiểm tra lớp và biên bằng ảnh.
- Các câu hỏi v1.0 số 1, 2, 4, 5, 6 chưa có ca ảnh đủ rõ để chốt; ghi rõ còn mở. Đây là một lý do giữ Hold, không suy diễn câu trả lời.
- Phân biệt nguyên nhân: YOLO26 thiếu lớp `three_wheeler` là giới hạn/mapping của máy; bẫy loose `img_035` lọt review là lỗi quy trình kiểm box. Overlay SAM cập nhật cho `img_013` và `img_002` có mask riêng cho phương tiện và người nhìn thấy; vùng tiếp xúc/che khuất vẫn cần duyệt đỉnh mask thủ công trước khi dùng làm nhãn polygon.

## 7. QC và quyết định (Part 6)

- Mẫu QC gồm 5/24 ảnh batch và 102 đối tượng; danh sách ở `qc/qc_sample.csv`. Mẫu lấy ngẫu nhiên theo seed 42, không có ảnh `can_review` được đưa thêm.
- Lượt QC ban đầu rà overlay trên ảnh gốc và ghi nhận 2 lỗi/2 ảnh: 2/5 = 40% ảnh lỗi; 2/102 = 2.0% đối tượng. Cụ thể: `img_031` box #445 gán `person` cho mũ trưng bày; `img_035` box motorbike #496 còn lỏng và trùng ca bẫy loose đã lọt. Ba ảnh còn lại trong mẫu (`img_003`, `img_006`, `img_014`) không ghi nhận lỗi rõ trong lượt rà đó. Đây là tỷ lệ của mẫu ban đầu, không phải ước lượng chắc chắn cho toàn bộ lô.
- Hai lỗi đã được sửa trong `labels/qc_fixed.json`, và `python scripts/build_release.py` đã tạo lại **release candidate**: `img_031` bỏ box #445; `img_035` khôi phục bbox #496 về `orig_box` trong `prelabel/traps_key.json`. Nhật ký từng sửa nằm ở `qc/qc_remediation.csv`; manifest ghi nhận hai ảnh được thay. Số lượng ứng viên sau sửa là 24 ảnh, 520 đối tượng (giảm một box false positive).
- `qc_report_auto.md` lưu thống kê phiếu QC ban đầu: cận trên Wilson 95% rộng; mẫu nhỏ không chứng minh ngưỡng lỗi 2%. Việc sửa hai phát hiện không biến kết quả QC ban đầu thành 0 lỗi. Cần lấy mẫu hậu sửa độc lập trước khi xem xét phát hành.
- Tỷ lệ bắt bẫy vẫn là 3/12 = 25%, thấp hơn yêu cầu 80%; năm câu hỏi guideline v1.0 vẫn mở. Các lý do này độc lập với hai sửa QC nên quyết định vẫn là HOLD.

**Quyết định: HOLD**, ngày 2026-11-18.

Release candidate đã được build để lưu kết quả sửa và phục vụ kiểm tra; đây chưa phải bản được phê duyệt phát hành. Cần rà độc lập sau sửa, xem lại 12 bẫy và chốt các câu hỏi guideline còn mở trước khi cân nhắc Release.

## 8. Ghi chú cá nhân

- Dùng AI để làm bài: Có sử dụng YOLO26 và YOLOE để tạo pre-label; sử dụng ChatGPT hỗ trợ đọc hướng dẫn, kiểm tra quy trình và phân tích kết quả threshold. Tôi kiểm tra lại nhãn Gold bằng cách tự gán thủ công trên CVAT.
- Gold có bị nhiễm không: Tôi đã trực tiếp gán nhãn thủ công 4 ảnh Gold trên CVAT. Chưa có xác nhận trong ghi chép rằng tôi không xem pre-label hoặc hỏi AI trước khi hoàn thành Gold, nên chưa thể kết luận Gold không bị nhiễm.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, PyTorch 2.15.0+cpu, không có CUDA GPU được nhận diện. Repo hướng dẫn CVAT 2.81.0; đây là phiên bản tham chiếu, chưa xác minh phiên bản thực tế trên máy. Vướng mắc cài đặt chưa được ghi nhận.
- Cache: hiện có `data/cache/yolo26m_all_c005.coco.json` và `data/cache/yoloe26s_three_wheeler_all_c005.coco.json`; chưa có nhật ký lệnh xác nhận những cache này có được dùng cho merge/threshold sweep hay không. Các run r01–r06 được ghi riêng trong `work/runs.csv`.
- Lỗi / bug script đã gặp: File `submission/labels/gold_manual.json` ban đầu chưa tồn tại. Sau khi export annotations từ CVAT định dạng COCO 1.0 và đổi tên file, đã validate thành công với 4 ảnh và 47 annotation (45 vật thể, 2 `ignore_region`).
- Một điều tôi làm khác lần sau: Tôi sẽ hoàn thiện và kiểm tra dữ liệu Gold trước khi chạy đánh giá threshold, đồng thời ghi lại thời gian và các quyết định sửa nhãn ngay trong quá trình review để tránh thiếu dữ liệu khi làm báo cáo.

---

### Phụ lục – Kết quả Threshold Calibration

Nguồn: `submission/qc/threshold_sweep.md`, `submission/qc/threshold_sweep.csv`, `submission/qc/threshold_sweep.png`.

- Số ảnh Gold: 4 ảnh.
- Tổng annotation Gold: 47, gồm 45 vật thể thuộc lớp đối tượng và 2 `ignore_region`.
- Mục tiêu Recall: 90%.
- Person: Recall tối đa 89%, giữ mặc định 0.3.
- Bicycle: 4 đối tượng Gold, giữ mặc định 0.3.
- Motorbike: Recall tối đa 78%, giữ mặc định 0.3.
- Car: Recall tối đa 88%, giữ mặc định 0.3.
- Bus: 0 đối tượng Gold, giữ mặc định 0.3.
- Truck: 0 đối tượng Gold, giữ mặc định 0.3.
- Three_wheeler: 6 đối tượng Gold, threshold 0.15 đạt Recall 100%, Precision 67%.

Tham số threshold đề xuất cho bước merge:

`--thr "three_wheeler=0.15"`

Lưu ý: Những lớp không được khai báo trong tham số `--thr` sử dụng confidence mặc định 0.3.
