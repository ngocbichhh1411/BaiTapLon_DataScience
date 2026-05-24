# Học phần: Bài tập lớn Khoa học dữ liệu (Data Science)
## Đề tài: Xây dựng mô hình học máy dự đoán giá xe ô tô dựa trên các thông số kỹ thuật

---

## I. GIỚI THIỆU ĐỀ TÀI
Dự án tập trung vào việc nghiên cứu và xây dựng các mô hình Học máy (Machine Learning) nhằm dự đoán giá bán của xe ô tô dựa trên tập hợp các thông số kỹ thuật cốt lõi (như dung tích động cơ, công suất tối đa, mô-men xoắn, hệ dẫn động, kích thước) và các yếu tố hao mòn vật lý (năm sản xuất, số kilomet đã đi). Kết quả dự án hướng tới việc cung cấp một giải pháp định giá khách quan, chính xác cho thị trường giao dịch xe ô tô.

* **Dữ liệu thô ban đầu:** `car details v4.csv` (gồm 20 trường thuộc tính kỹ thuật và giá bán của các dòng xe).
* **Môi trường phát triển:** Python (Pandas, NumPy, Scikit-learn), MySQL Server, VS Code / Jupyter Notebook.

--
---

## III. QUY TRÌNH TRIỂN KHAI DỰ ÁN

### 1. Quản trị dữ liệu với MySQL
* Toàn bộ tệp dữ liệu thô ban đầu được nạp hệ thống lưu trữ tập trung thông qua công cụ **Table Data Import Wizard** của MySQL Workbench, giúp đồng bộ cấu trúc bảng và quản lý thuộc tính an toàn, bảo mật giữa các thành viên.

### 2. Tiền xử lý và Lọc dữ liệu 
Chương trình Python thực hiện gột rửa các lỗi sai lệch dữ liệu:
* **Chuẩn hóa dữ liệu định lượng:** Sử dụng hàm cắt chuỗi và biểu thức chính quy (Regex) để loại bỏ các ký tự nhiễu văn bản (xóa chữ `cc` ở cột `Engine`, chữ `bhp` ở cột `Max_Power`, lọc chuỗi số ở cột `Max_Torque`). Ép kiểu dữ liệu về dạng số thực (`float`).
* **Xử lý giá trị khuyết thiếu (Missing Values):** Tự động điền các ô trống dạng Số bằng giá trị **Trung vị (Median)** của cột nhằm loại bỏ ảnh hưởng của giá trị ngoại lai; điền ô trống dạng Chữ bằng giá trị **Yếu vị (Mode)** phổ biến nhất.
* **Khử trùng lặp:** Quét và xóa bỏ triệt để các bản ghi lặp lại 100% nhằm bảo đảm tính khách quan cho mô hình.

### 3. Phân tích khám phá (EDA) & Mã hóa thuộc tính
* Thực hiện vẽ các biểu đồ phân phối giá xe, ma trận tương quan giữa các thông số kỹ thuật để tìm ra các thuộc tính ảnh hưởng lớn nhất đến giá bán.
* Sử dụng các kỹ thuật mã hóa dữ liệu (Label Encoding/One-Hot Encoding) để chuyển đổi các trường thông tin dạng chữ thành các ma trận số học phù hợp cho máy tính xử lý.

### 4. Huấn luyện và Đánh giá mô hình
* Chia tập dữ liệu thành các phần Train/Test để tiến hành thử nghiệm trên nhiều thuật toán Học máy khác nhau (như Tuyến tính - Linear Regression, Cây quyết định - Decision Tree, Rừng ngẫu nhiên - Random Forest).
* Đánh giá hiệu năng và lựa chọn ra mô hình tối ưu nhất dựa trên các chỉ số đo lường sai số.

---

## IV. CẤU TRÚC THƯ MỤC DỰ ÁN
```text
├── car details v4.csv          # Tệp dữ liệu thô ban đầu
├── locdulieu.py                # Mã nguồn Python thực hiện lọc dữ liệu
├── car_details_cleaned.csv    # Sản phẩm dữ liệu sạch sau tiền xử lý
├── [Các file code EDA/Model]   # Các file code phân tích và mô hình của thành viên khác
└── README.md                   # Tài liệu hướng dẫn chung của dự án
