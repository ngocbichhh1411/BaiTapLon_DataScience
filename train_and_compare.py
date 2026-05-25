# =========================================================
# GIAI ĐOẠN 2 - HUẤN LUYỆN, SO SÁNH VÀ PHÂN TÍCH MÔ HÌNH
# File: train_and_compare.py
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Thư viện lưu mô hình tối ưu theo cấu trúc mới

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # Thêm thuật toán XGBoost mới cập nhật
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Cài đặt font chữ cho biểu đồ để không bị lỗi tiếng Việt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ MỚI
print("===== 1. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ MỚI =====")
# Đọc chính xác 2 file dữ liệu final mới cập nhật của bạn
train_df = pd.read_csv("train_car_final (1).csv")
test_df = pd.read_csv("test_car_final (1).csv")
print("✅ Nạp dữ liệu tập Train và tập Test mới thành công!")

target_col = "Price_log" 

# Tách biến độc lập (X) và biến mục tiêu (y)
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

print(f"📊 Kích thước tập Train: {X_train.shape[0]:,} dòng, {X_train.shape[1]} cột")
print(f"📊 Kích thước tập Test:  {X_test.shape[0]:,} dòng, {X_test.shape[1]} cột")

# 2. KHỞI TẠO CÁC MÔ HÌNH THEO QUY TRÌNH CẬP NHẬT
models = {
    "Linear Regression (Tuyến tính)": LinearRegression(),
    "Random Forest (Rừng ngẫu nhiên)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost (Boosting nâng cao)": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
}

results = []

# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
print("\n===== 2. TIẾN HÀNH HUẤN LUYỆN VÀ CHẤM ĐIỂM =====")
for name, model in models.items():
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    # Dự đoán giá xe trên tập Test
    y_pred_log = model.predict(X_test)
    
    # Đổi ngược giá trị Logarit về tiền thật để đo lường sai số thực tế
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred_log)
    
    # Tính toán các chỉ số đánh giá
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)
    
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Model_Object": model
    })
    print(f"  -> Chạy xong mô hình: {name}")

# In bảng thống kê kết quả trực quan ra màn hình terminal
df_results = pd.DataFrame(results)
print("\n📊 BẢNG KẾT QUẢ SO SÁNH HIỆU NĂNG MÔ HÌNH:")
print(df_results[["Model", "MAE", "R2 Score"]].to_string(index=False))

# --- BIỂU ĐỒ 1: SO SÁNH ĐỘ CHÍNH XÁC (R2 SCORE) ---
plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="R2 Score", data=df_results, palette="Blues_r")
plt.title("SO SÁNH ĐỘ CHÍNH XÁC (R2 SCORE) GIỮA CÁC MÔ HÌNH", fontsize=14, fontweight='bold', pad=15)
plt.ylim(0, 1.1)
for i, val in enumerate(df_results["R2 Score"]):
    plt.text(i, val + 0.02, f"{val:.4f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# 4. PHÂN TÍCH SÂU VÀ LƯU MÔ HÌNH CHIẾN THẮNG (XGBOOST)
print("\n===== 3. XUẤT MÔ HÌNH VÀ PHÂN TÍCH ĐỘ QUAN TRỌNG COÓT TRỌNG =====")
best_model_name = "XGBoost (Boosting nâng cao)"
xgb_model = models[best_model_name]

# Lưu trữ mô hình tối ưu bằng joblib theo đúng cấu trúc của nhóm bạn
joblib.dump(xgb_model, "car_price_model.pkl")
print(f"💾 Đã lưu cấu trúc mô hình tối ưu nhất vào file 'car_price_model.pkl'!")

# Tính toán mức độ đóng góp của từng biến số vào kết quả định giá từ XGBoost
importances = xgb_model.feature_importances_
df_importance = pd.DataFrame({
    'Yếu tố': X_train.columns,
    'Độ quan trọng': importances
}).sort_values(by='Độ quan trọng', ascending=False)

print("\n📈 ĐỘ QUAN TRỌNG CỦA CÁC YẾU TỐ ĐỊNH GIÁ XE (TOP 10):")
print(df_importance.head(10).to_string(index=False))

# --- BIỂU ĐỒ 2: ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Độ quan trọng', y='Yếu tố', data=df_importance.head(10), palette='viridis')
plt.title('MỨC ĐỘ ẢNH HƯỞNG CỦA CÁC YẾU TỐ ĐẾN GIÁ XE (XGBOOST)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Tỷ lệ % đóng góp quyết định', fontsize=12)
plt.tight_layout()
plt.show()