import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# 1. Cấu hình tiêu đề và giao diện trang web
st.set_page_config(
    page_title="Dự Đoán Giá Xe Ô Tô",
    page_icon="🚗",
    layout="wide"
)

# 2. Tải mô hình và bộ giải mã mã hóa (Dùng cache để chạy mượt)
@st.cache_resource
def load_models_and_encoders():
    # Tải mô hình XGBoost
    model = joblib.load("car_price_model.pkl")
    # Tải bộ Target Encoder
    with open("target_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_models_and_encoders()
    global_mean = encoders['global_mean']
    load_success = True
except Exception as e:
    st.error(f"❌ Lỗi: Không tìm thấy file! Hãy chắc chắn 'car_price_model.pkl' và 'target_encoders.pkl' nằm cùng một thư mục với file app.py.")
    load_success = False

# 3. Xây dựng giao diện nếu tải file thành công
if load_success:
    st.title("🚗 HỆ THỐNG ĐỊNH GIÁ & DỰ ĐOÁN GIÁ XE Ô TÔ")
    st.write("Vui lòng nhập các thông số của xe bên dưới để nhận giá dự đoán từ mô hình học máy XGBoost.")
    st.markdown("---")
    
    # Chia giao diện làm 2 cột cho đẹp mắt
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Thông tin phân loại xe")
        
        # 1. Lấy sơ đồ Hãng -> Dòng xe đã được tích hợp trong file pkl
        car_mapping = encoders.get('car_mapping', {})
        
        # 2. Tạo danh sách Hãng xe
        list_make = sorted(list(car_mapping.keys())) if car_mapping else sorted(list(encoders['Make'].keys()))
        make = st.selectbox("Hãng xe (Make):", list_make)
        
        # 3. Tự động lọc Dòng xe tương ứng với Hãng vừa chọn
        if car_mapping and make in car_mapping:
            list_model_filtered = sorted(car_mapping[make])
        else:
            list_model_filtered = sorted(list(encoders['Model'].keys())) # Fallback dự phòng
            
        model_car = st.selectbox("Dòng xe (Model):", list_model_filtered)
        
        # --- Các thông số bên dưới giữ nguyên ---
        list_fuel = sorted(list(encoders['Fuel Type'].keys()))
        list_transmission = sorted(list(encoders['Transmission'].keys()))
        list_owner = sorted(list(encoders['Owner'].keys()))
        list_seller = sorted(list(encoders['Seller Type'].keys()))
        
        fuel_type = st.selectbox("Loại nhiên liệu (Fuel Type):", list_fuel)
        transmission = st.selectbox("Hộp số (Transmission):", list_transmission)
        owner = st.selectbox("Số đời chủ (Owner):", list_owner)
        seller_type = st.selectbox("Người bán (Seller Type):", list_seller)

    with col2:
        st.subheader("⚙️ Thông số kỹ thuật & Kích thước")
        
        # Các ô nhập số liệu dựa trên các trường tính toán từ file "train_car_final.csv" của ông
        year = st.number_input("Năm sản xuất:", min_value=1980, max_value=2026, value=2020, step=1)
        kilometer = st.number_input("Số Km đã đi (Kilometer):", min_value=0, value=40000, step=1000)
        engine_cc = st.number_input("Dung tích động cơ (Engine CC):", min_value=100.0, value=1500.0, step=50.0)
        max_power = st.number_input("Công suất tối đa (Max_Power_bhp):", min_value=10.0, value=120.0, step=5.0)
        max_torque = st.number_input("Mô-men xoắn (Max_Torque_Nm):", min_value=10.0, value=170.0, step=5.0)
        
        length = st.number_input("Chiều dài xe (mm):", min_value=1000.0, value=4000.0, step=10.0)
        width = st.number_input("Chiều rộng xe (mm):", min_value=1000.0, value=1750.0, step=10.0)
        height = st.number_input("Chiều cao xe (mm):", min_value=1000.0, value=1500.0, step=10.0)
        
        # Logic tính Car_Age = 2024 - Year (theo đúng file tiền xử lý của ông)
        car_age = 2024 - year

    st.markdown("---")
    
    # Nút bấm kích hoạt dự đoán
    if st.button("💰 TÍNH GIÁ XE", type="primary", use_container_width=True):
        
        # Bước A: Mã hóa Target Encoding cho các biến chữ nhập vào bằng dữ liệu trong dict encoders
        encoded_input = {
            'Make': encoders['Make'].get(make, global_mean),
            'Model': encoders['Model'].get(model_car, global_mean),
            'Fuel Type': encoders['Fuel Type'].get(fuel_type, global_mean),
            'Transmission': encoders['Transmission'].get(transmission, global_mean),
            'Owner': encoders['Owner'].get(owner, global_mean),
            'Seller Type': encoders['Seller Type'].get(seller_type, global_mean),
            'Kilometer': kilometer,
            'Length': length,
            'Width': width,
            'Height': height,
            'Max_Power_bhp': max_power,
            'Max_Torque_Nm': max_torque,
            'Engine_CC': engine_cc,
            'Car_Age': car_age
        }
        
        # Chuyển thành DataFrame
        input_df = pd.DataFrame([encoded_input])
        
        # Bước B: Đồng bộ hóa thứ tự cột khớp 100% với mô hình ban đầu nhờ feature_names_in_
        try:
            model_features = model.feature_names_in_
            input_df = input_df[model_features]
            
            # Bước C: Mô hình dự đoán (kết quả trả về là dạng log: Price_log)
            pred_log = model.predict(input_df)[0]
            
            # Bước D: Nghịch đảo logarit (dùng expm1 theo đúng logic huấn luyện của ông)
            predicted_price = np.expm1(pred_log)
            
            # Hiển thị kết quả ra màn hình với định dạng tiền tệ đẹp mắt
            st.success("🎉 Dự đoán thành công!")
            st.metric(label="GIÁ XE ƯỚC TÍNH (Đơn vị tiền tệ của dữ liệu gốc)", value=f"${predicted_price:,.2f}")
            
            if car_age < 0:
                st.warning(f"⚠️ Lưu ý: Đời xe ({year}) mới hơn mốc tính toán năm 2024 của dữ liệu gốc.")
                
        except Exception as pred_error:
            st.error(f"Lỗi khi xử lý dữ liệu đầu vào hoặc dự đoán: {pred_error}")