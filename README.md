# Parkinson's Telemonitoring Modeling Project

## Mục tiêu
Dự án này tập trung vào:
- Khám phá bộ dữ liệu Parkinson's Telemonitoring
- Tiền xử lý và kỹ thuật đặc trưng để giảm đa cộng tuyến
- So sánh hai cách xây dựng mô hình:
  1. Một mô hình đa đầu ra dự đoán cùng lúc `motor_UPDRS` và `total_UPDRS`
  2. Hai mô hình độc lập dự đoán riêng từng biến mục tiêu
- Lưu model tốt nhất và dự đoán dữ liệu mới từ bàn phím

## File chính

- `EDA_documentation.md`: Tóm tắt các bước EDA đã làm và các bước đề xuất tiếp theo.
- `full_EDA.ipynb`: Notebook tổng hợp EDA từ cả hai nguồn.
- `feature_engineer.py`: Module feature engineering chính.
- `feature_engineering_documentation.md`: Giải thích ý nghĩa, cách dùng từng hàm trong module.
- `model_survey.ipynb`: Notebook training, đánh giá, so sánh mô hình.
- `predict_input.py`: Script cho phép nhập dữ liệu mới từ bàn phím và dự đoán.
- `models/best_model.pt`: Model tốt nhất được lưu lại sau khi chạy `model_survey.ipynb`.

## File dữ liệu

- `parkinsons+telemonitoring/parkinsons_updrs.data`: File gốc chứa 5.875 bản ghi.
- `preprocessed_data.csv`: Dữ liệu đã xử lý sơ bộ từ EDA hiện có.
- `ucimlrepo` được sử dụng để tải dữ liệu trực tiếp từ UCI repository.

## Cách dùng

### 1. Chạy EDA
- Mở `full_EDA.ipynb` để xem các bước kiểm tra dữ liệu, khám phá correlation, outlier, phân phối và PCA.
- `EDA_documentation.md` chứa tóm tắt chi tiết.

### 2. Chạy training và so sánh mô hình
- Mở `model_survey.ipynb` và chạy từng cell.
- Notebook sẽ:
  1. Chuẩn bị dữ liệu trực tiếp từ UCI repository bằng `ucimlrepo`
  2. Chia tập train/test theo `subject#`
  3. Huấn luyện cả mô hình multi-output và hai mô hình riêng biệt
  4. So sánh RMSE/R² giữa các mô hình
  5. Lưu model tốt nhất vào `models/best_model.pt`

> Lưu ý: cần cài đặt `ucimlrepo` nếu chưa có:
> ```bash
> pip install ucimlrepo
> ```

### 3. Dự đoán với dữ liệu nhập tay
- Chạy script:
```bash
python predict_input.py
```
- Nhập các giá trị đặc trưng theo yêu cầu.
- Script sẽ tải model đã lưu và in ra `motor_UPDRS` và `total_UPDRS`.

## Hướng dẫn mở rộng

- Nếu muốn thử thêm feature engineering, sửa `feature_engineer.py` để thêm các đặc trưng tương tác, log-transform hoặc one-hot encoding.
- Nếu muốn thử thêm mô hình, bổ sung trong `model_survey.ipynb` hoặc module `model_pipeline.py`.
- Nếu muốn dùng pipeline mô hình khác, hãy lưu lại một object tuân thủ phương thức `predict()`.
