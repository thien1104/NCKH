import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Đường dẫn
MODEL_PATH = r"Model_LSTM_Optimized.h5"
SCALER_X_PATH = r"scaler_X.pkl"
SCALER_Y_PATH = r"scaler_y.pkl"
NEW_DATA_PATH = r"Book1.xlsx"
OLD_DATA_PATH = r"data_1723.csv"

# Load mô hình và scaler
lstm_model = load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# Đọc dữ liệu
df_new = pd.read_excel(NEW_DATA_PATH, sheet_name='KetQua')
df_old = pd.read_csv(OLD_DATA_PATH)

if df_new.empty:
    st.error("File dữ liệu mới không có dữ liệu!")
    st.stop()

def find_lags(x_value, df_old):
    closest_row = df_old.iloc[(df_old['X'] - x_value).abs().argsort()[:1]]
    if closest_row.empty:
        return np.nan, np.nan, np.nan
    
    lag_1 = closest_row['Q2'].values[0]
    lag_2 = df_old[df_old.index == closest_row.index[0] - 1]['Q2'].values[0] if closest_row.index[0] > 0 else lag_1
    rolling_mean = df_old[df_old.index <= closest_row.index[0]]['Q2'].rolling(window=3).mean().values[-1]
    
    return lag_1, lag_2, rolling_mean

df_new.loc[0, ['lag_1', 'lag_2', 'rolling_mean']] = find_lags(df_new.loc[0, 'X'], df_old)

y_new_pred = []
for i in range(len(df_new)):
    X_input = df_new.loc[i, ['X', 'lag_1', 'lag_2', 'rolling_mean']].values.reshape(1, -1)
    X_input_scaled = scaler_X.transform(X_input).reshape((1, X_input.shape[1], 1))
    
    y_pred_scaled = lstm_model.predict(X_input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    y_pred = max(y_pred, 0)
    y_new_pred.append(y_pred)
    
    if i + 1 < len(df_new):
        df_new.loc[i + 1, 'lag_1'] = y_pred
        df_new.loc[i + 1, 'lag_2'] = df_new.loc[i, 'lag_1']
        df_new.loc[i + 1, 'rolling_mean'] = df_new.loc[max(0, i - 1):i + 1, 'lag_1'].mean()

df_new['Q2_Predicted'] = y_new_pred

# Hiển thị kết quả trên Streamlit
st.set_page_config(page_title="Dự đoán lưu lượng mưa", layout="wide")
st.markdown("""<h1 style='text-align: center; color: purple;'>Dự đoán lưu lượng mưa Q2 bằng Long Short Term Memory (LSTM)</h1>""", unsafe_allow_html=True)

# Định dạng cột ngày nếu có
if 'Date' in df_new.columns:
    df_new['Date'] = pd.to_datetime(df_new['Date']).dt.strftime('%d-%m-%Y')
    df_new = df_new.sort_values(by='Date')

st.write("### 🔴 Dữ liệu dự đoán từ mô hình LSTM.")
st.dataframe(df_new[['Date', 'X', 'Q2_Predicted']].rename(columns={'Date':'Ngày', 'X': 'Lượng mưa (mm)', 'Q2_Predicted': 'Dự đoán Q2 (m³/s)'}))

# Biểu đồ
st.markdown("### 🔴 Kết quả dự đoán lưu lượng Q2 theo ngày")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_new['Date'], df_new['Q2_Predicted'], label="Q2 dự báo (m³/s)", marker='o')
ax.set_xlabel("Ngày", fontsize=14)
ax.set_ylabel("Lưu lượng Q2 (m³/s)", fontsize=14)
ax.legend()
ax.grid()
st.pyplot(fig)
