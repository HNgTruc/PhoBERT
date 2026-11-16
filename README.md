# Phân loại tin nhắn SMS lừa đảo và quảng cáo tiếng Việt bằng PhoBERT

Mã nguồn thử nghiệm sơ bộ của đề tài NCKH sinh viên, Nhóm 5, môn Phương pháp nghiên cứu khoa học trong CNTT (841324), Khoa CNTT, Trường Đại học Sài Gòn.

Bài toán: phân loại một tin SMS tiếng Việt vào 3 nhãn `LUA_DAO` (lừa đảo), `QUANG_CAO` (quảng cáo), `BINH_THUONG` (bình thường). So sánh ba mô hình:

| Mô hình | Vai trò | Tên trên Hugging Face |
| --- | --- | --- |
| TF-IDF + Logistic Regression | Mô hình cơ sở | (scikit-learn) |
| PhoBERT-base | Mô hình chính | `vinai/phobert-base` |
| ViSoBERT | Mô hình so sánh | `uitnlp/visobert` |

## Cấu trúc thư mục

```
src/gan_nhan_so_bo.py                    Tải bộ dữ liệu, bỏ trùng, gợi ý nhãn, chọn 600 tin đợt 1, tạo mẫu kappa
notebooks/thu-nghiem-so-bo.ipynb         Huấn luyện và đánh giá 3 mô hình (5-fold), tính Cohen's kappa
du-lieu/processed/tran-2026/
    full_dataset.csv                     Bộ dữ liệu gốc (đã ẩn danh bởi tác giả bộ dữ liệu)
    gan-nhan-so-bo.csv                   Dữ liệu đã gán nhãn, cột nhan_chot là nhãn người duyệt chốt
    mau-kappa-100.csv                    100 tin do người gán thứ hai gán độc lập (cột nhan)
ket-qua/ket-qua-so-bo.json               Kết quả lần chạy ngày 08/11/2026: cấu hình, phiên bản thư viện, số liệu từng fold
requirements.txt                         Thư viện cần cài khi chạy trên máy cá nhân
```

## Dữ liệu

Nguồn: bộ `trannguyenthaituan/vietnamese_sms_dataset` trên Hugging Face, commit `a90e2bd7e8df2376939658075ba8094dcee488ad`, giấy phép CC BY 4.0. Bộ gốc có 2.991 dòng với 2 nhãn (0 hợp lệ, 1 rác/lừa đảo), còn 2.629 tin sau khi bỏ trùng. Số điện thoại, số tiền đã được thay bằng thẻ `[PHONE]`, `[MONEY]`...; script quét thêm số điện thoại và dãy số dài còn sót.

Thử nghiệm dùng 600 tin đợt 1 (200 tin mỗi nhãn gợi ý, seed 42). Bỏ 3 tin chưa có nhãn chốt hoặc gán `KHONG_RO`, còn 597 tin.

## Cách chạy

### Bước 1 (không bắt buộc): tạo lại dữ liệu gán nhãn sơ bộ

Chỉ cần thư viện chuẩn của Python. Chạy từ thư mục gốc của gói này:

```
python src/gan_nhan_so_bo.py
```

Script tải `full_dataset.csv` nếu chưa có, rồi ghi `gan-nhan-so-bo.csv` và `mau-kappa-100.csv`. Lưu ý: chạy lại sẽ ghi đè hai file này và xóa cột `nhan_chot`, `nhan` mà người duyệt đã điền. Muốn chạy thử thì sao lưu thư mục `du-lieu/` trước.

### Bước 2: chạy notebook

Cách khuyến nghị là Google Colab với GPU T4:

1. Mở `notebooks/thu-nghiem-so-bo.ipynb` trên Colab, chọn Runtime > Change runtime type > T4 GPU.
2. Chạy lần lượt từng ô. Ô "Tải dữ liệu" sẽ hỏi file: tải lên `du-lieu/processed/tran-2026/gan-nhan-so-bo.csv`.
3. Ô Cohen's kappa ở cuối hỏi thêm file `mau-kappa-100.csv`.
4. Ô "Lưu kết quả" ghi file JSON giống `ket-qua/ket-qua-so-bo.json`.

Thời gian: mô hình cơ sở vài giây; PhoBERT và ViSoBERT mỗi mô hình khoảng 5 đến 10 phút trên T4.

Chạy trên máy cá nhân: cài thư viện trong `requirements.txt`, cài Java (VnCoreNLP cần Java để tách từ cho PhoBERT), chép `gan-nhan-so-bo.csv` và `mau-kappa-100.csv` vào cùng thư mục với notebook, và sửa đường dẫn `/content/vncorenlp` trong ô tách từ thành một thư mục trên máy. Nếu VnCoreNLP không chạy được, notebook tự chuyển sang `underthesea` và in thông báo.

## Cấu hình thử nghiệm

| Tham số | Giá trị |
| --- | --- |
| Seed | 42 |
| Chia dữ liệu | 5-fold phân tầng theo nhãn |
| Epochs, batch, learning rate | 4, 16, 2e-5 |
| Độ dài tối đa | 128 token |
| Tách từ cho PhoBERT | RDRSegmenter (VnCoreNLP) |
| Phiên bản đã chạy | torch 2.14.1+cu130, transformers 5.22.1, scikit-learn 1.6.1, pandas 2.2.3 |

Mỗi mô hình được đánh giá trên 4 kiểu tập kiểm tra: `goc` (nguyên văn), `bo_dau` (bỏ dấu tiếng Việt), `chen_ky_tu` (biến dạng khoảng 20% số từ như kẻ gửi rác hay làm), `xoa_the` (xóa thẻ QC/TB để xem mô hình có chỉ học thẻ hay không).

## Kết quả sơ bộ (trung bình ± độ lệch chuẩn qua 5 fold, %)

| Mô hình | Kiểu tập kiểm tra | Macro-F1 | F1 lớp lừa đảo |
| --- | --- | --- | --- |
| TF-IDF + LR | goc | 91,8 ± 2,3 | 91,9 ± 3,4 |
| TF-IDF + LR | bo_dau | 89,5 ± 3,0 | 88,5 ± 3,9 |
| TF-IDF + LR | chen_ky_tu | 90,7 ± 2,3 | 91,7 ± 3,0 |
| TF-IDF + LR | xoa_the | 91,6 ± 2,2 | 91,7 ± 3,0 |
| PhoBERT-base | goc | 87,6 ± 2,9 | 90,8 ± 2,9 |
| PhoBERT-base | bo_dau | 82,8 ± 2,4 | 82,2 ± 5,3 |
| PhoBERT-base | chen_ky_tu | 78,4 ± 2,0 | 84,9 ± 3,1 |
| PhoBERT-base | xoa_the | 85,3 ± 4,1 | 89,1 ± 3,5 |
| ViSoBERT | goc | 89,7 ± 3,3 | 92,7 ± 4,0 |
| ViSoBERT | bo_dau | 85,1 ± 3,0 | 86,5 ± 8,3 |
| ViSoBERT | chen_ky_tu | 86,9 ± 4,0 | 89,2 ± 2,7 |
| ViSoBERT | xoa_the | 84,7 ± 2,2 | 90,2 ± 3,4 |

Với 597 tin, PhoBERT-base chưa vượt mô hình cơ sở TF-IDF + LR, nên giả thuyết H1 chưa được ủng hộ. PhoBERT giảm nhiều hơn ViSoBERT khi tin bị bỏ dấu (H2 đúng chiều) nhưng mức chênh chưa ổn định giữa các fold. Đây là kết quả sơ bộ trên mẫu nhỏ, chi tiết từng fold và ma trận nhầm lẫn nằm trong `ket-qua/ket-qua-so-bo.json`.

## Ghi nguồn

Bộ dữ liệu: trannguyenthaituan, "vietnamese_sms_dataset", Hugging Face, CC BY 4.0.
PhoBERT: D. Q. Nguyen và A. T. Nguyen, "PhoBERT: Pre-trained language models for Vietnamese", Findings of EMNLP 2020.
