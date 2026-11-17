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

- Thời gian gán tay: chưa ghi nhận. Thời gian duyệt batch: chưa ghi nhận. `submission/time_log.csv` hiện chỉ có header, chưa có dòng dữ liệu.
- Thời gian inference trong `model_log.csv` không thay thế thời gian gán tay/duyệt: YOLO26 pilot 0.74–0.75 giây/ảnh, YOLOE pilot 0.34 giây/ảnh, SAM 2.1 B 5.93 giây/ảnh; đây là số trên CPU.
- Nguồn: `submission/time_log.csv`, `submission/model_log.csv`.
- Thiên lệch: Tôi đã gán nhãn Gold trước nên có thể đã quen với hình ảnh và đặc điểm đối tượng. Điều này có thể ảnh hưởng đến tính khách quan khi đánh giá pre-label.
- Kết luận: chưa đủ dữ liệu thời gian người để kết luận AI có tiết kiệm công hay không. Threshold sweep gợi ý `three_wheeler=0.15` đạt Recall 100% nhưng Precision 67% trên 6 vật Gold; đây là ước lượng nhỏ và có nguy cơ lạc quan do chọn/nghiệm thu trên cùng Gold.

Ba quy tắc giảm thiên kiến:

1. Kiểm tra toàn bộ ảnh thay vì chỉ kiểm tra những box AI đã dự đoán.
2. Không mặc định chấp nhận box chỉ vì confidence cao; phải kiểm tra vị trí, kích thước và class.
3. Ghi lại mọi trường hợp sửa, xóa và thêm box để có số liệu đánh giá AI error profile.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hóa? | Lỗi quan sát | File trong masks/ |
|---|---|---|---|---|---|
| `img_013` tuk-tuk | SAM 2.1 B (`sam2.1_b.pt`) | 6 (4 dương, 2 âm) | Có; mask gồm xe, người lái và hành khách | Mask gom nhiều cá thể thành một vùng; cần tách person khỏi xe theo contract | `masks/tuktuk_img013.jpg` |
| `img_001` pickup | SAM 2.1 B (`sam2.1_b.pt`) | 6 (4 dương, 2 âm) | Không thấy người lái/hàng hóa trong vùng mask | Biên xe khá gọn, nền đường bên ngoài được loại; chưa đo IoU | `masks/pickup_img001.jpg` |
| `img_002` motorbike | SAM 2.1 B (`sam2.1_b.pt`) | 7 (4 dương, 3 âm) | Có; mask gồm cả người lái và xe máy | Click âm chưa tách được người khỏi xe; một mask đang gộp hai đối tượng | `masks/motorbike_img002.jpg` |

- Đối chiếu guideline mục 4.2: pickup cho thấy biên có thể bám theo vật; hai mask tuk-tuk và motorbike còn gộp người với xe, trong khi contract yêu cầu vật chạm nhau có polygon riêng. Cần thêm click âm/điều chỉnh prompt để tách, rồi kiểm tra lỗ và biên ở mức khoảng 3 px.
- Interactor SAM trong CVAT: chưa có ghi chép cho một lần chạy bên trong CVAT; ba ảnh trên được tạo bằng `scripts/sam2_points.py` để mô phỏng click.

## 6. Calibration (Part 6)

- Số quyết định: 0; `calibration_decisions.csv` hiện chỉ có header.
- Quyết định quan trọng nhất: chưa có `CAL-xx` được ghi.
- Quy tắc mới: chưa có. `guideline_v1.1_calibrated.md` vẫn là khung chưa điền, còn các trường `<<...>>`.
- Nguyên nhân gốc: chưa thể phân loại thành guideline mơ hồ, lỗi người gán hay lỗi máy khi chưa có quyết định calibration và ví dụ ca khó.

## 7. QC và quyết định (Part 6)

- Mẫu QC Part 6: chưa tạo; `qc_sample.csv` và `qc_review.csv` hiện không có trong workspace. Part 4 đã review 24 ảnh batch và tính review stats; việc đó không thay thế mẫu QC cuối.
- Người kiểm: Tôi. Hạn chế: Tôi vừa là người gán nhãn vừa là người duyệt nên có nguy cơ thiên lệch.
- Lỗi theo đối tượng và theo ảnh: chưa tính; chưa có mẫu QC hoặc `qc_report_auto.md`, nên chưa thể so với ngưỡng Release ≤ 2%.
- Top lỗi từ QC: chưa thống kê. Các ví dụ định tính đã thấy trong pre-label (tuk-tuk thành `car`, xe đạp thành `three_wheeler`) không thay thế kết quả QC trên nhãn cuối.
- Cận trên 95% cho mẫu nhỏ: chưa tính vì chưa có mẫu QC.
- Script có thể kiểm tra cấu trúc, tính nhất quán, hình học/nhãn theo quy tắc đã lập trình và tính thống kê khi có dữ liệu đầu vào; không thể tự xác nhận một nhãn đúng về ngữ nghĩa nếu thiếu người đối chiếu ảnh.

**Quyết định: HOLD**, ngày 2026-11-17.

Lý do: Part 4 đã có batch review và kết quả bẫy, nhưng tỷ lệ bắt bẫy 25% thấp hơn ngưỡng 80%. Calibration chưa có quyết định, guideline v1.1 còn là mẫu, và mẫu QC/`qc_report_auto.md` chưa được tạo; do đó chưa có cơ sở Release.

Việc cần làm tiếp:
- Part 5 có ba ảnh mask; chỉnh lại tuk-tuk và motorbike để tách người khỏi xe theo contract.
- Ghi thời gian review batch vào `time_log.csv` nếu có số đo thật.
- Ghi ít nhất 3 quyết định calibration, cập nhật guideline v1.1, rồi tạo mẫu QC và báo cáo.
- Kiểm tra các lỗi theo đối tượng và theo ảnh.
- Chỉ quyết định Release khi đáp ứng điều kiện chất lượng của dự án.

## 8. Ghi chú cá nhân

- Dùng AI để làm bài: Có sử dụng YOLO26 và YOLOE để tạo pre-label; sử dụng ChatGPT hỗ trợ đọc hướng dẫn, kiểm tra quy trình và phân tích kết quả threshold. Tôi kiểm tra lại nhãn Gold bằng cách tự gán thủ công trên CVAT.
- Gold có bị nhiễm không: Tôi đã trực tiếp gán nhãn thủ công 4 ảnh Gold trên CVAT. Chưa có xác nhận trong ghi chép rằng tôi không xem pre-label hoặc hỏi AI trước khi hoàn thành Gold, nên chưa thể kết luận Gold không bị nhiễm.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: các lần chạy trong `model_log.csv` đều ghi thiết bị CPU; không có thông tin GPU, cấu hình máy hoặc phiên bản CVAT. Việc dùng CVAT để gán Gold đã được ghi ở trên; vướng mắc cài đặt chưa được ghi nhận.
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
