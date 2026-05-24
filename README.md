# Học phần: Bài tập lớn Khoa học dữ liệu
## Đề tài: Xây dựng mô hình học máy dự đoán giá xe ô tô dựa trên các thông số kỹ thuật

---

## PHẦN VIỆC: SQL VÀ LỌC DỮ LIỆU (DATA CLEANING)

---

### 1. Quản trị dữ liệu với MySQL
* Hệ quản trị cơ sở dữ liệu sử dụng: **MySQL** (thông qua MySQL Workbench).
* Phương thức nạp dữ liệu: Sử dụng công cụ trực quan `Table Data Import Wizard` để ánh xạ trực tiếp tệp dữ liệu gốc `car details v4.csv` thành cấu trúc bảng quan hệ, tối ưu hóa việc lưu trữ tập trung và quản lý thuộc tính.

### 2. Bộ lọc dữ liệu tự động với Python (`locdulieu.py`)
Mã nguồn Python được thiết kế để kết nối, gột rửa dữ liệu thô thông qua hai thư viện chuyên dụng là **Pandas** và **NumPy**. Các tác vụ cốt lõi đã hoàn thành bao gồm:
* **Chuẩn hóa dữ liệu định lượng:** Loại bỏ ký tự nhiễu văn bản (cắt bỏ đơn vị `cc` ở cột `Engine`, `bhp` ở cột `Max_Power`) và bóc tách chuỗi số phức tạp ở cột `Max_Torque` bằng biểu thức chính quy (Regex). Ép kiểu dữ liệu về dạng số thực (`float`).
* **Xử lý giá trị khuyết thiếu (Missing Values):** * Tự động tính toán và điền các ô trống dạng Số bằng giá trị **Trung vị (Median)** của cột để tránh hiện tượng lệch dữ liệu do các giá trị ngoại lai (Outliers).
  * Điền các ô trống dạng Chữ (như cột `Drivetrain`) bằng giá trị **Yếu vị (Mode)** - giá trị xuất hiện phổ biến nhất.
* **Loại bỏ dữ liệu trùng lặp:** Sử dụng hàm loại bỏ trùng lặp để quét và xóa triệt để các bản ghi bị lặp 100%, bảo đảm tính khách quan cho mô hình.

### 3. Thành phẩm đầu ra
* Xuất thành công tệp dữ liệu hoàn toàn sạch: **`car_details_cleaned.csv`**.
* Tỷ lệ ô trống ở tất cả 20 thuộc tính được đưa về mức **0%**.
* Tập dữ liệu đạt độ chuẩn hóa tối đa, sẵn sàng bàn giao cho các công đoạn Mã hóa thuộc tính (Encoding) và Huấn luyện mô hình học máy tiếp theo.
