import pandas as pd
import numpy as np

# 1. Đọc file dữ liệu vào Python (Hãy đảm bảo file 'car details v4.csv' nằm cùng thư mục này)
df = pd.read_csv('car details v4.csv')
print("--- Dữ liệu thô ban đầu ---")
print(df.head())

# 2. Làm sạch cột 'Engine' (Xóa chữ 'cc' để chuyển về dạng số)
if 'Engine' in df.columns:
    df['Engine'] = df['Engine'].str.replace(' cc', '', regex=False)
    df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')

# 3. Làm sạch cột 'Max_Power' (Xóa chữ 'bhp' để chuyển về dạng số)
if 'Max_Power' in df.columns:
    df['Max_Power'] = df['Max_Power'].str.replace(' bhp', '', regex=False)
    df['Max_Power'] = pd.to_numeric(df['Max_Power'], errors='coerce')

# 4. Làm sạch cột 'Max_Torque' (Xóa các ký tự chữ và đơn vị phức tạp)
if 'Max_Torque' in df.columns:
    df['Max_Torque'] = df['Max_Torque'].str.extract(r'(\d+\.?\d*)').astype(float)

# 5. Xử lý dữ liệu bị khuyết thiếu (Missing Values)
print("\nSố lượng ô trống trước khi xử lý:")
print(df.isnull().sum())

# Điền giá trị trung vị (Median) cho các cột dạng số
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Điền giá trị xuất hiện nhiều nhất cho cột dạng chữ
cat_cols = df.select_dtypes(include=[object]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

# 6. Loại bỏ dữ liệu trùng lặp (Duplicate rows)
df = df.drop_duplicates()

print("\n--- Dữ liệu sau khi làm sạch hoàn toàn ---")
print(df.head())

# 7. Xuất file dữ liệu sạch ra một file CSV mới
df.to_csv('car_details_cleaned.csv', index=False)
print("\n Đã xuất file 'car_details_cleaned.csv' thành công!")