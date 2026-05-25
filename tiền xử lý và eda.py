# =========================================================
# THÀNH VIÊN 2 - DATA PREPROCESSING, EDA & FEATURE ENGINEERING
# Đề tài: Dự đoán giá xe ô tô
# Dataset: car_details_cleaned.csv (Version 4)
# =========================================================

import sys
try:
    # Hỗ trợ in tiếng Việt không bị lỗi font trên console Windows
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Cài đặt font chữ cho biểu đồ để không bị lỗi ô vuông
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. ĐỌC DỮ LIỆU & LÀM SẠCH BƯỚC ĐẦU
# =========================================================
df = pd.read_csv("car_details_cleaned.csv")
print("===== 1. KÍCH THƯỚC DỮ LIỆU BAN ĐẦU =====")
print(f"Số dòng: {df.shape[0]:,} | Số cột: {df.shape[1]}")

# Xóa dòng trùng lặp
df = df.drop_duplicates()

# Hàm trích xuất số bằng Regex (Bắt chuẩn xác các thông số như '87 bhp @ 6000 rpm' -> 87.0)
def extract_num(text):
    if pd.isna(text): return np.nan
    nums = re.findall(r"(\d+\.?\d*)", str(text))
    return float(nums[0]) if nums else np.nan

# Áp dụng Regex để lấy phần số từ các cột kỹ thuật
df["Max_Power_bhp"] = df["Max Power"].apply(extract_num)
df["Max_Torque_Nm"] = df["Max Torque"].apply(extract_num)
df["Engine_CC"] = df["Engine"].apply(extract_num) 

# Điền khuyết nhanh bằng trung vị (median)
for col in ["Max_Power_bhp", "Max_Torque_Nm", "Engine_CC", "Kilometer"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Tính tuổi xe (Giả sử năm hiện tại là 2024)
df["Car_Age"] = 2024 - df["Year"]

# Xóa các cột text cũ không cần thiết hoặc dễ gây nhiễu
cols_to_drop = [
    "Model", "Year", "Max Power", "Max Torque", "Engine", 
    "Length", "Width", "Height", "Location", "Color", 
    "Fuel Tank Capacity", "Drivetrain", "Seating Capacity"
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

# =========================================================
# 2. KHÁM PHÁ DỮ LIỆU (EDA)
# =========================================================
print("\n===== 2. THỐNG KÊ PRICE & KILOMETER =====")
print(f"💰 Price (Giá xe): Min = {df['Price'].min():,.0f} | Max = {df['Price'].max():,.0f}")
print(f"🚗 Kilometer (Số km đã đi): Min = {df['Kilometer'].min():,.0f} | Max = {df['Kilometer'].max():,.0f}")

# --- 2.1 VẼ BOXPLOT ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot(df['Price'].dropna(), vert=True)
axes[0].set_title('Boxplot Price (Phát hiện xe giá siêu cao)', fontweight='bold')
axes[0].set_ylabel('Price (Rupee)')

axes[1].boxplot(df['Kilometer'].dropna(), vert=True)
axes[1].set_title('Boxplot Kilometer (Phát hiện xe chạy quá nhiều/ít)', fontweight='bold')
axes[1].set_ylabel('Kilometer')
plt.tight_layout()
plt.show()

# --- 2.2 VẼ PHÂN BỔ GIÁ XE ---
bins_price = [0, 200000, 500000, 1000000, 2000000, 5000000, 10000000, float('inf')]
labels_price = ['< 200k', '200k - 500k', '500k - 1M', '1M - 2M', '2M - 5M', '5M - 10M', '> 10M']

df['Price_Group'] = pd.cut(df['Price'], bins=bins_price, labels=labels_price, include_lowest=True)
group_price = df['Price_Group'].value_counts().reindex(labels_price).fillna(0)

plt.figure(figsize=(14, 6))
ax_price = sns.barplot(x=group_price.index, y=group_price.values, palette='Blues_d')
plt.title('PHÂN BỔ GIÁ XE TOÀN BỘ DATASET', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Khoảng giá', fontsize=14, fontweight='bold')
plt.ylabel('Số lượng xe', fontsize=14, fontweight='bold')

for i, v in enumerate(group_price.values):
    if v > 0:
        ax_price.text(i, v + (max(group_price.values)*0.01), f'{int(v):,}', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()
df.drop(columns=['Price_Group'], inplace=True) 

# --- 2.3 VẼ 4 BIỂU ĐỒ SO SÁNH KILOMETER ---
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# 1. Tổng quan
bins1 = [0, 1000, 50000, 100000, 150000, 200000, 300000, 600000, float('inf')]
labels1 = ['<1k', '1k-50k', '50k-100k', '100k-150k', '150k-200k', '200k-300k', '300k-600k', '>600k']
df['Km_Group'] = pd.cut(df['Kilometer'], bins=bins1, labels=labels1, include_lowest=True)
group1 = df['Km_Group'].value_counts().reindex(labels1).fillna(0)
sns.barplot(x=group1.index, y=group1.values, ax=axes[0,0], palette='viridis')
axes[0,0].set_title('1. TỔNG QUAN PHÂN BỔ SỐ KM', fontsize=16, fontweight='bold')

# 2. Thấp bất thường (<1000 km)
df_low = df[df['Kilometer'] < 1000]
bins2 = list(range(0, 1100, 100))
labels2 = [str(i) for i in range(100, 1100, 100)]
group2 = pd.cut(df_low['Kilometer'], bins=bins2, labels=labels2, include_lowest=True).value_counts().reindex(labels2).fillna(0)
sns.barplot(x=group2.index, y=group2.values, ax=axes[0,1], palette='flare')
axes[0,1].set_title('2. VÙNG KM THẤP BẤT THƯỜNG (< 1,000 km)', fontsize=16, fontweight='bold')

# 3. Cao bất thường (300k - 600k km)
df_high = df[(df['Kilometer'] > 300000) & (df['Kilometer'] <= 600000)]
bins3 = [300000, 400000, 500000, 600000]
labels3 = ['300k-400k', '400k-500k', '500k-600k']
group3 = pd.cut(df_high['Kilometer'], bins=bins3, labels=labels3, include_lowest=True).value_counts().reindex(labels3).fillna(0)
sns.barplot(x=group3.index, y=group3.values, ax=axes[1,0], palette='YlOrBr')
axes[1,0].set_title('3. VÙNG KM CAO BẤT THƯỜNG (300k - 600k km)', fontsize=16, fontweight='bold')

# 4. Siêu cao (> 1 Triệu km)
df_1m = df[df['Kilometer'] > 1000000]
if len(df_1m) > 0:
    bins4 = [1000000, 1500000, 2000000, float('inf')]
    labels4 = ['1M-1.5M', '1.5M-2M', '>2M']
    group4 = pd.cut(df_1m['Kilometer'], bins=bins4, labels=labels4, include_lowest=True).value_counts().reindex(labels4).fillna(0)
    sns.barplot(x=group4.index, y=group4.values, ax=axes[1,1], palette='Reds')
    axes[1,1].set_title('4. VÙNG SIÊU CAO (> 1 Triệu km)', fontsize=16, fontweight='bold')
else:
    axes[1,1].text(0.5, 0.5, "Không có xe nào > 1 Triệu km", ha='center', fontsize=15)
    axes[1,1].set_title('4. VÙNG SIÊU CAO (> 1 Triệu km)', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
df.drop(columns=['Km_Group'], inplace=True)

# =========================================================
# 3. XỬ LÝ OUTLIER BẰNG IQR (LỌC BỎ DÒNG NGOẠI LAI)
# =========================================================
print("\n===== 3. XỬ LÝ OUTLIER =====")
initial_rows = df.shape[0]

# Các cột cần kiểm tra và xóa outlier
num_cols = ["Kilometer", "Engine_CC", "Max_Power_bhp", "Max_Torque_Nm", "Car_Age"]
num_cols = [c for c in num_cols if c in df.columns]

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Lọc giữ lại các dòng nằm trong vùng an toàn
    df = df[(df[col] >= lower) & (df[col] <= upper)]

final_rows = df.shape[0]

# In báo cáo kết quả lọc Outlier
print(f"✅ Đã lọc xong ngoại lai!")
print(f"👉 Số dòng bị loại: {initial_rows - final_rows:,} dòng")
print(f"📉 Giá xe (Price): {df['Price'].min():,.0f} → {df['Price'].max():,.0f}")
print(f"📏 Số km (Kilometer): {df['Kilometer'].min():,.0f} km → {df['Kilometer'].max():,.0f} km")
print(f"📊 Còn lại: {final_rows:,} dòng")

# =========================================================
# 4. CHIA TẬP TRAIN / TEST TRƯỚC KHI MÃ HÓA
# =========================================================
print("\n===== 4. CHIA TẬP TRAIN / TEST =====")
# Biến đổi Logarit biến mục tiêu Price để phân phối chuẩn hơn
y = np.log1p(df["Price"])
X = df.drop(columns=["Price"])

# CHIA TRAIN/TEST TRƯỚC ĐỂ TRÁNH RÒ RỈ DỮ LIỆU (DATA LEAKAGE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Đã chia tập Train/Test thành công!")

# =========================================================
# 5. TARGET ENCODING (MÃ HÓA TRÁNH RÒ RỈ DỮ LIỆU)
# =========================================================
print("\n===== 5. MÃ HÓA DỮ LIỆU =====")
encoding_dict = {}

# Mượn tạm y_train ghép vào X_train để tính giá trung bình
train_temp = X_train.copy()
train_temp['Price_Log'] = y_train

cat_features = ['Make', 'Fuel Type', 'Seller Type', 'Transmission', 'Owner']
cat_features = [c for c in cat_features if c in X_train.columns]

global_mean = train_temp['Price_Log'].mean()
encoding_dict['global_mean_log'] = global_mean

for col in cat_features:
    # Lập bản đồ tính trung bình giá trị cho từng nhóm
    mean_map = train_temp.groupby(col)['Price_Log'].mean().to_dict()
    encoding_dict[col] = mean_map
    
    # Áp dụng lên Train và Test
    X_train[col] = X_train[col].map(mean_map).fillna(global_mean)
    X_test[col] = X_test[col].map(mean_map).fillna(global_mean)

# Xuất từ điển ra file .pkl
with open('target_encoders.pkl', 'wb') as f:
    pickle.dump(encoding_dict, f)

print("✅ Đã mã hóa dữ liệu tránh Leakage và lưu từ điển thành công!")
print(f"📊 Tập Train: {X_train.shape[0]:,} dòng (đã lưu vào 'train_car_final.csv')")
print(f"📊 Tập Test:  {X_test.shape[0]:,} dòng (đã lưu vào 'test_car_final.csv')")

# =========================================================
# 6. FEATURE SCALING (CHUẨN HÓA DỮ LIỆU)
# =========================================================
print("\n===== 6. FEATURE SCALING =====")
scaler = StandardScaler()

scale_cols = num_cols + cat_features

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print("✅ Đã chuẩn hóa xong tập dữ liệu bằng StandardScaler.")

# =========================================================
# 7. XUẤT FILE CHO MÔ HÌNH HỌC MÁY
# =========================================================
train_df = X_train.copy()
train_df["Price_log"] = y_train.values

test_df = X_test.copy()
test_df["Price_log"] = y_test.values

train_df.to_csv("train_car_final.csv", index=False)
test_df.to_csv("test_car_final.csv", index=False)

print("\n🚀 HOÀN TẤT! Dữ liệu đã sẵn sàng cho bước Train Model.")