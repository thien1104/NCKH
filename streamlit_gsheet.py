import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Dự đoán lưu lượng", layout="wide")

# Tạo kết nối đến Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# Đọc dữ liệu từ Google Sheets
df = conn.read()

# Kiểm tra nếu dữ liệu có tồn tại
if df is not None and not df.empty:
    # Chuyển đổi cột 'Day' sang kiểu datetime
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")

    # Nếu có giá trị NaT (lỗi khi chuyển đổi), loại bỏ hàng đó
    df = df.dropna(subset=["Day"])

    # Chuyển đổi ngày thành định dạng "dd/MM"
    df["Day"] = df["Day"].dt.strftime("%d/%m")

    # Tiêu đề lớn
    st.markdown(
        """<h1 style='text-align: center; color: purple; font-size:60px;'>Dự đoán lưu lượng mưa trên sông A Lưới</h1>""",
        unsafe_allow_html=True,
    )

    # Biểu đồ 1: Lượng mưa
    st.markdown(
        """<h2 style='text-align: center; color: red; font-size:42px;'>📊 Biểu đồ lượng mưa theo ngày</h2>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 5])
    with col1:
        st.markdown("<h3 style='color: blue;'> Chọn ngày:</h3>", unsafe_allow_html=True)
        selected_day = st.selectbox("", df["Day"].unique(), key="day_x")
        selected_data = df[df["Day"] == selected_day]

        if not selected_data.empty:
            X_value = selected_data["X"].values[0]
            Q2_value = selected_data["Q2"].values[0]
            st.markdown(
                f"<h2>➡ Lượng mưa (X): <span style='color: red;'>{X_value} mm</span></23>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h2>➡ Lưu lượng dự đoán (Q2): <span style='color: red;'>{Q2_value} m³/s</span></h2>",
                unsafe_allow_html=True,
            )

    with col2:
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        ax1.bar(df["Day"], df["X"], color="blue", alpha=0.7)
        ax1.set_xlabel("Ngày", fontsize=12)
        ax1.set_ylabel("Lượng mưa (mm)", fontsize=12)
        ax1.tick_params(axis="both", labelsize=10)
        st.pyplot(fig1)

    # Biểu đồ 2: Lưu lượng dự đoán
    st.markdown(
        """<h2 style='text-align: center; color: red; font-size:42px;'>📈 Biểu đồ lưu lượng dự đoán theo ngày</h2>""",
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns([2, 5])
    with col3:
        st.empty()
    with col4:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(df["Day"], df["Q2"], marker="o", linestyle="-", color="red")
        ax2.set_xlabel("Ngày", fontsize=12)
        ax2.set_ylabel("Lưu lượng dự đoán (m³/s)", fontsize=12)
        ax2.tick_params(axis="both", labelsize=10)
        st.pyplot(fig2)

else:
    st.write("Không có dữ liệu để hiển thị.")
