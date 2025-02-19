import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Thiết lập giao diện rộng
st.set_page_config(layout="wide")

# Tiêu đề ứng dụng
st.markdown("""
    <h1 style='text-align: center; color: purple;'>Dự đoán giá trị X và Q2 sử dụng Random Forest</h1>
""", unsafe_allow_html=True)

# Đọc dữ liệu
file_path = r'data_1723.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Không tìm thấy file dữ liệu!")
    st.stop()

# Chuyển đổi định dạng cột ngày
df['Day'] = pd.to_datetime(df['Day'], format='%m/%d/%Y')
df['Day_num'] = df['Day'].map(pd.Timestamp.toordinal)

# Chuẩn bị dữ liệu đầu vào
X = df[['Day_num', 'X']].values
y_q2 = df['Q2'].values.reshape(-1, 1)
y_x = df['X'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y_q2 = MinMaxScaler()
scaler_y_x = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_q2_scaled = scaler_y_q2.fit_transform(y_q2)
y_x_scaled = scaler_y_x.fit_transform(y_x)

X_train, X_test, y_q2_train, y_q2_test = train_test_split(X_scaled, y_q2_scaled, test_size=0.2, random_state=42)
X_train_x, X_test_x, y_x_train, y_x_test = train_test_split(X_scaled, y_x_scaled, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest cho Q2
rf_model_q2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_q2.fit(X_train, y_q2_train.ravel())

# Huấn luyện mô hình Random Forest cho X
rf_model_x = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_x.fit(X_train_x, y_x_train.ravel())

# Dự đoán
y_q2_pred_scaled = rf_model_q2.predict(X_test)
y_q2_pred = scaler_y_q2.inverse_transform(y_q2_pred_scaled.reshape(-1, 1))
y_q2_test_original = scaler_y_q2.inverse_transform(y_q2_test)

y_x_pred_scaled = rf_model_x.predict(X_test_x)
y_x_pred = scaler_y_x.inverse_transform(y_x_pred_scaled.reshape(-1, 1))
y_x_test_original = scaler_y_x.inverse_transform(y_x_test)

# Tính toán các chỉ số đánh giá
mae_q2 = mean_absolute_error(y_q2_test_original, y_q2_pred)
rmse_q2 = np.sqrt(mean_squared_error(y_q2_test_original, y_q2_pred))
nse_q2 = 1 - (np.sum((y_q2_pred - y_q2_test_original) ** 2) / np.sum((y_q2_test_original - np.mean(y_q2_test_original)) ** 2))

# Hiển thị kết quả đánh giá
st.write("## Đánh giá mô hình")
st.markdown(f"""
### - **MAE (Q2) :** <span style='color: red;'>{mae_q2:.2f}</span>  
### - **RMSE (Q2):** <span style='color: red;'>{rmse_q2:.2f}</span>  
### - **NSE (Q2) :** <span style='color: red;'>{nse_q2:.2f}</span>
""", unsafe_allow_html=True)    

# Vẽ biểu đồ dự đoán
st.write("### So sánh giữa giá trị thực tế và dự đoán")
col1, col2 = st.columns([3, 2])

with col1:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_q2_test_original, label="Q2 thực tế", color='blue')
    ax.plot(y_q2_pred, label="Q2 dự đoán", color='orange', linestyle='dashed')
    ax.set_xlabel("Mẫu dữ liệu")
    ax.set_ylabel("Giá trị Q2")
    ax.legend()
    st.pyplot(fig)

with col2:
    # Chọn ngày để dự đoán Q2
    selected_date = st.date_input("***Chọn ngày để dự đoán :***", pd.to_datetime('today'))
    selected_day_num = pd.Timestamp(selected_date).toordinal()

# Dự đoán giá trị Q2 cho ngày được chọn
    input_data = np.array([[selected_day_num, 0]])  # X = 0 vì không sử dụng giá trị X
    input_data_scaled = scaler_X.transform(input_data)
    predicted_q2_scaled = rf_model_q2.predict(input_data_scaled)
    predicted_q2 = scaler_y_q2.inverse_transform(predicted_q2_scaled.reshape(-1, 1))

    st.markdown(f"<h3>Giá trị dự đoán Q2 cho ngày {selected_date}: <span style='color: red;'>{predicted_q2[0][0]:.2f}</span></h3>", unsafe_allow_html=True)

# Hiển thị bảng so sánh giá trị thực tế và dự đoán
st.write("### Bảng so sánh giá trị thực tế và dự đoán")
st.dataframe(pd.DataFrame({
    "Thực tế Q2": y_q2_test_original.flatten(),
    "Dự đoán Q2": y_q2_pred.flatten(),
    "Thực tế X": y_x_test_original.flatten(),
    "Dự đoán X": y_x_pred.flatten()
}), use_container_width=True)
