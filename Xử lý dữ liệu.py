import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. ĐỌC DỮ LIỆU
df = pd.read_csv("car_details_cleaned.csv")

# 2. FEATURE ENGINEERING
def extract_num(text):
    nums = re.findall(r"(\d+\.?\d*)", str(text))
    return float(nums[0]) if nums else 0.0

df["Max_Power_bhp"] = df["Max Power"].apply(extract_num)
df["Max_Torque_Nm"] = df["Max Torque"].apply(extract_num)
df["Car_Age"] = 2024 - df["Year"]

# Xóa cột không cần
df.drop(columns=["Model", "Year", "Max Power", "Max Torque"], inplace=True)

# 3. OUTLIER HANDLING (TRƯỚC SPLIT - THEO Ý M)
num_cols = [
    "Kilometer", "Engine", "Max_Power_bhp",
    "Max_Torque_Nm", "Length", "Width",
    "Height", "Car_Age"
]

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# 4. CHIA TRAIN / TEST
y = np.log1p(df["Price"])
X = df.drop(columns=["Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. MÃ HÓA (ONE-HOT)
cat_cols = X_train.select_dtypes(include="object").columns

X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# đảm bảo cột giống nhau
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 6. CHUẨN HÓA (CHỈ NUMERIC THỰC SỰ)
scale_cols = [
    "Kilometer", "Engine", "Max_Power_bhp",
    "Max_Torque_Nm", "Length", "Width",
    "Height", "Car_Age"
]

# chỉ giữ cột tồn tại (tránh lỗi)
scale_cols = [col for col in scale_cols if col in X_train.columns]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# 7. XUẤT FILE
train_df = X_train.copy()
train_df["Price_log"] = y_train.values

test_df = X_test.copy()
test_df["Price_log"] = y_test.values

train_df.to_csv("train_car.csv", index=False)
test_df.to_csv("test_car.csv", index=False)

print("DONE - xử lý xong dữ liệu")