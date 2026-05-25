# =========================================================
# THÀNH VIÊN 2 - DATA PREPROCESSING, EDA & FEATURE ENGINEERING
# Đề tài: Dự đoán giá xe ô tô
# Dataset: car_details_cleaned.csv
# =========================================================

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# CÀI ĐẶT HIỂN THỊ
# =========================================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. ĐỌC DỮ LIỆU
# =========================================================

df_raw = pd.read_csv("car_details_cleaned.csv")
df = df_raw.copy()

print("===== 1. THÔNG TIN DỮ LIỆU =====")

print(f"Số dòng: {df.shape[0]:,}")
print(f"Số cột : {df.shape[1]}")

# Xóa dòng trùng lặp
df = df.drop_duplicates()

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

# Hàm trích xuất số
def extract_num(text):

    if pd.isna(text):
        return np.nan

    nums = re.findall(r"(\d+\.?\d*)", str(text))

    return float(nums[0]) if nums else np.nan

# Trích xuất dữ liệu số
df["Max_Power_bhp"] = df["Max Power"].apply(extract_num)
df["Max_Torque_Nm"] = df["Max Torque"].apply(extract_num)
df["Engine_CC"] = df["Engine"].apply(extract_num)

# Rút gọn Model để tránh overfitting
df["Model"] = df["Model"].apply(
    lambda x: str(x).split()[0] if pd.notna(x) else np.nan
)

# Tạo tuổi xe
df["Car_Age"] = 2024 - df["Year"]

# =========================================================
# 3. KHÁM PHÁ DỮ LIỆU (EDA)
# =========================================================

print("\n===== 2. THỐNG KÊ DỮ LIỆU =====")

print(f"""
💰 Price:
Min = {df['Price'].min():,.0f}
Max = {df['Price'].max():,.0f}

🚗 Kilometer:
Min = {df['Kilometer'].min():,.0f}
Max = {df['Kilometer'].max():,.0f}
""")

# =========================================================
# BOXPLOT
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].boxplot(df['Price'].dropna())
axes[0].set_title('Boxplot Price')

axes[1].boxplot(df['Kilometer'].dropna())
axes[1].set_title('Boxplot Kilometer')

plt.tight_layout()
plt.show()

# =========================================================
# PHÂN BỐ GIÁ XE
# =========================================================

bins_price = [
    0,
    200000,
    500000,
    1000000,
    2000000,
    5000000,
    10000000,
    float('inf')
]

labels_price = [
    '<200k',
    '200k-500k',
    '500k-1M',
    '1M-2M',
    '2M-5M',
    '5M-10M',
    '>10M'
]

df['Price_Group'] = pd.cut(
    df['Price'],
    bins=bins_price,
    labels=labels_price,
    include_lowest=True
)

group_price = (
    df['Price_Group']
    .value_counts()
    .reindex(labels_price)
    .fillna(0)
)

plt.figure(figsize=(14, 6))

ax = sns.barplot(
    x=group_price.index,
    y=group_price.values,
    palette='Blues_d'
)

plt.title(
    'PHÂN BỔ GIÁ XE TOÀN BỘ DATASET',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel('Khoảng giá')
plt.ylabel('Số lượng xe')

for i, v in enumerate(group_price.values):

    if v > 0:

        ax.text(
            i,
            v + max(group_price.values) * 0.01,
            f'{int(v):,}',
            ha='center',
            fontweight='bold'
        )

plt.tight_layout()
plt.show()

# =========================================================
# PHÂN TÍCH CHI TIẾT KILOMETER
# =========================================================

df_plot2 = df_raw.copy()

df_plot2['Kilometer'] = pd.to_numeric(
    df_plot2['Kilometer'],
    errors='coerce'
)

df_plot2 = df_plot2.dropna(subset=['Kilometer'])

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# =========================================================
# 1. TỔNG QUAN PHÂN BỔ KM
# =========================================================

bins1 = [
    0,
    1000,
    50000,
    100000,
    150000,
    200000,
    300000,
    600000,
    float('inf')
]

labels1 = [
    '<1k km',
    '1k-50k',
    '50k-100k',
    '100k-150k',
    '150k-200k',
    '200k-300k',
    '300k-600k',
    '>600k'
]

df_plot2['Km_Group'] = pd.cut(
    df_plot2['Kilometer'],
    bins=bins1,
    labels=labels1,
    include_lowest=True
)

group1 = (
    df_plot2['Km_Group']
    .value_counts()
    .reindex(labels1)
    .fillna(0)
)

ax1 = axes[0, 0]

sns.barplot(
    x=group1.index,
    y=group1.values,
    ax=ax1,
    palette='viridis'
)

ax1.set_title(
    '1. TỔNG QUAN PHÂN BỔ SỐ KM',
    fontsize=16,
    fontweight='bold'
)

for i, v in enumerate(group1.values):

    if v > 0:

        ax1.text(
            i,
            v + max(group1.values) * 0.02,
            f'{int(v):,}',
            ha='center',
            fontweight='bold'
        )

# =========================================================
# 2. KM THẤP BẤT THƯỜNG (<1000 km)
# =========================================================

df_low = df_plot2[
    df_plot2['Kilometer'] < 1000
]

bins2 = list(range(0, 1100, 100))

labels2 = [
    str(i)
    for i in range(100, 1100, 100)
]

group2 = pd.cut(
    df_low['Kilometer'],
    bins=bins2,
    labels=labels2,
    include_lowest=True
).value_counts().reindex(labels2).fillna(0)

ax2 = axes[0, 1]

sns.barplot(
    x=group2.index,
    y=group2.values,
    ax=ax2,
    palette='flare'
)

ax2.set_title(
    '2. KM THẤP BẤT THƯỜNG (<1000 km)',
    fontsize=16,
    fontweight='bold'
)

for i, v in enumerate(group2.values):

    if v > 0:

        ax2.text(
            i,
            v + 1,
            f'{int(v):,}',
            ha='center',
            fontweight='bold'
        )

# =========================================================
# 3. KM CAO BẤT THƯỜNG (300k - 600k)
# =========================================================

df_high = df_plot2[
    (df_plot2['Kilometer'] > 300000) &
    (df_plot2['Kilometer'] <= 600000)
]

bins3 = [
    300000,
    400000,
    500000,
    600000
]

labels3 = [
    '300k-400k',
    '400k-500k',
    '500k-600k'
]

group3 = pd.cut(
    df_high['Kilometer'],
    bins=bins3,
    labels=labels3,
    include_lowest=True
).value_counts().reindex(labels3).fillna(0)

ax3 = axes[1, 0]

sns.barplot(
    x=group3.index,
    y=group3.values,
    ax=ax3,
    palette='YlOrBr'
)

ax3.set_title(
    f'3. KM CAO BẤT THƯỜNG ({len(df_high):,} xe)',
    fontsize=16,
    fontweight='bold'
)

for i, v in enumerate(group3.values):

    if v > 0:

        ax3.text(
            i,
            v + 0.3,
            f'{int(v):,}',
            ha='center',
            fontweight='bold'
        )

# =========================================================
# 4. XE > 1 TRIỆU KM
# =========================================================

df_1m = df_plot2[
    df_plot2['Kilometer'] > 1_000_000
]

ax4 = axes[1, 1]

if len(df_1m) > 0:

    bins4 = [
        1000000,
        1500000,
        2000000,
        float('inf')
    ]

    labels4 = [
        '1M-1.5M',
        '1.5M-2M',
        '>2M'
    ]

    group4 = pd.cut(
        df_1m['Kilometer'],
        bins=bins4,
        labels=labels4,
        include_lowest=True
    ).value_counts().reindex(labels4).fillna(0)

    sns.barplot(
        x=group4.index,
        y=group4.values,
        ax=ax4,
        palette='Reds'
    )

    for i, v in enumerate(group4.values):

        if v > 0:

            ax4.text(
                i,
                v + 0.3,
                f'{int(v):,}',
                ha='center',
                fontweight='bold'
            )

else:

    ax4.text(
        0.5,
        0.5,
        'Không có xe > 1 triệu km',
        ha='center',
        fontsize=14
    )

ax4.set_title(
    f'4. XE > 1 TRIỆU KM ({len(df_1m):,} xe)',
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# Xóa cột tạm
df.drop(columns=['Price_Group'], inplace=True)

# =========================================================
# 4. CHIA TRAIN / TEST
# =========================================================

print("\n===== 3. CHIA TRAIN / TEST =====")

# Log Transform Price
y = np.log1p(df["Price"])

# Bỏ cột Price khỏi X
X = df.drop(columns=["Price"])

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Đã chia Train/Test thành công!")

# =========================================================
# 5. XỬ LÝ OUTLIER (CHỈ TRAIN)
# =========================================================

print("\n===== 4. XỬ LÝ OUTLIER =====")

num_cols = [
    "Kilometer",
    "Engine_CC",
    "Max_Power_bhp",
    "Max_Torque_Nm",
    "Car_Age"
]

num_cols = [
    c for c in num_cols
    if c in X_train.columns
]

initial_rows = X_train.shape[0]

# Lọc outlier theo IQR
for col in num_cols:

    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (
        (X_train[col] >= lower) &
        (X_train[col] <= upper)
    )

    X_train = X_train[mask]
    y_train = y_train[mask]

final_rows = X_train.shape[0]

# =========================================================
# THỐNG KÊ SAU KHI LỌC
# =========================================================

price_after = np.expm1(y_train)

price_min = price_after.min()
price_max = price_after.max()

km_min = X_train["Kilometer"].min()
km_max = X_train["Kilometer"].max()

print("✅ Đã lọc xong ngoại lai!")

print(
    f"👉 Số dòng bị loại: "
    f"{initial_rows - final_rows:,} dòng"
)

print(
    f"📉 Giá xe: "
    f"{price_min:,.0f}$ → {price_max:,.0f}$"
)

print(
    f"📏 Số km: "
    f"{km_min:,.0f} km → {km_max:,.0f} km"
)

print(
    f"📊 Còn lại: "
    f"{final_rows:,} dòng"
)

# =========================================================
# 6. TARGET ENCODING
# =========================================================

print("\n===== 5. TARGET ENCODING =====")

encoding_dict = {}

train_temp = X_train.copy()
train_temp['Price_log'] = y_train

cat_features = [
    'Make',
    'Model',
    'Fuel Type',
    'Seller Type',
    'Transmission',
    'Owner'
]

cat_features = [
    c for c in cat_features
    if c in X_train.columns
]

global_mean = train_temp['Price_log'].mean()

encoding_dict['global_mean'] = global_mean

for col in cat_features:

    mean_map = (
        train_temp
        .groupby(col)['Price_log']
        .mean()
        .to_dict()
    )

    encoding_dict[col] = mean_map

    X_train[col] = (
        X_train[col]
        .map(mean_map)
        .fillna(global_mean)
    )

    X_test[col] = (
        X_test[col]
        .map(mean_map)
        .fillna(global_mean)
    )

# Lưu encoder
with open("target_encoders.pkl", "wb") as f:
    pickle.dump(encoding_dict, f)

print("✅ Đã Target Encoding thành công!")

# =========================================================
# 7. XÓA CỘT KHÔNG CẦN THIẾT
# =========================================================

print("\n===== 6. XÓA CỘT KHÔNG CẦN THIẾT =====")

drop_cols = [
    "Year",
    "Max Power",
    "Max Torque",
    "Engine",
    "Location",
    "Color",
    "Drivetrain",
    "Fuel Tank Capacity",
    "Seating Capacity"
]

drop_cols = [
    c for c in drop_cols
    if c in X_train.columns
]

X_train.drop(columns=drop_cols, inplace=True)
X_test.drop(columns=drop_cols, inplace=True)

print("✅ Đã loại bỏ các cột không cần thiết!")

# =========================================================
# 8. XUẤT FILE
# =========================================================

train_df = X_train.copy()
train_df["Price_log"] = y_train.values

test_df = X_test.copy()
test_df["Price_log"] = y_test.values

train_df.to_csv("train_car_final.csv", index=False)
test_df.to_csv("test_car_final.csv", index=False)

print("\n===== HOÀN TẤT =====")

print(f"📊 Train: {train_df.shape[0]:,} dòng")
print(f"📊 Test : {test_df.shape[0]:,} dòng")

print("\n✅ Đã xuất:")
print(" - train_car_final.csv")
print(" - test_car_final.csv")
print(" - target_encoders.pkl")