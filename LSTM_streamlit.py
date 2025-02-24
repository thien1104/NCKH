import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Đọc dữ liệu
file_path = r'data_1723.csv'
df = pd.read_csv(file_path)

# Tạo đặc trưng mới (Feature Engineering)
df['lag_1'] = df['Q2'].shift(1)
df['lag_2'] = df['Q2'].shift(2)
df['rolling_mean'] = df['X'].rolling(window=3).mean().shift(1)
df.dropna(inplace=True)

# Định dạng cột ngày tháng
df['Day'] = pd.to_datetime(df['Day'], format='%m/%d/%Y')

# Chuẩn hóa dữ liệu
X = df[['X', 'lag_1', 'lag_2', 'rolling_mean']].values
y = df['Q2'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape dữ liệu cho LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Xây dựng mô hình LSTM
lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Compile mô hình
lstm_model.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình
history = lstm_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Dự đoán
y_pred_scaled = lstm_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_pred = np.maximum(y_pred, 0)
y_test_original = scaler_y.inverse_transform(y_test)

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Dự đoán lưu lượng - LSTM", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: purple;'>Dự đoán giá trị Q2 sử dụng LSTM</h1>
""", unsafe_allow_html=True)

# Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)
nse = 1 - np.sum((y_test_original - y_pred) ** 2) / np.sum((y_test_original - np.mean(y_test_original)) ** 2)

st.markdown("## 🔴 Kết quả đánh giá mô hình")
st.markdown(f"""
<h3>➡ MAE (Q2): <span style='color: red;'>{mae:.2f} m³/s</span>;
    RMSE (Q2): <span style='color: red;'>{rmse:.2f} m³/s</span>;
    NSE (Q2): <span style='color: red;'>{nse:.2f}</span></h3>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 4])
with col2:
    # Biểu đồ so sánh giá trị thực tế và dự đoán
    st.markdown("### 📈 Biểu đồ so sánh giữa giá trị thực tế và dự đoán (LSTM)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_original, label="Giá trị thực tế (m³/s)", color='blue')
    ax.plot(y_pred, label="Giá trị dự đoán (m³/s)", color='orange', linestyle='dashed')
    ax.set_xlabel("Mẫu dữ liệu")
    ax.set_ylabel("Giá trị Q2 (m³/s)")
    ax.legend()
    st.pyplot(fig)

with col1:
    # Hiển thị bảng dữ liệu so sánh
    data_comparison = pd.DataFrame({
    "Giá trị thực tế (m³/s)": y_test_original.flatten(),
    "Giá trị dự đoán (m³/s)": y_pred.flatten()
    })

    st.markdown("## 🔴 Bảng so sánh giá trị thực tế và dự đoán")
    st.dataframe(data_comparison)
# Thêm chức năng chọn ngày và dự đoán giá trị Q2
st.markdown("## 🔴 Dự đoán Q2 theo ngày")
selected_date = st.date_input("📅 Chọn ngày", min_value=df['Day'].min().date(), max_value=df['Day'].max().date())

if selected_date in df['Day'].dt.date.values:
    selected_row = df[df['Day'].dt.date == selected_date]
    input_features = selected_row[['X', 'lag_1', 'lag_2', 'rolling_mean']].values
    input_scaled = scaler_X.transform(input_features)
    input_scaled = input_scaled.reshape((1, input_scaled.shape[1], 1))
    prediction_scaled = lstm_model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
    st.markdown(f"### ➡ Giá trị thực tế Q2: {selected_row['Q2'].values[0]:.2f} m³/s")
    st.markdown(f"### ➡ Giá trị dự đoán Q2: {prediction:.2f} m³/s")
else:
    st.warning("Ngày được chọn không có trong tập dữ liệu.")


