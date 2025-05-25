import streamlit as st
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml import PipelineModel
from utils import get_spark_session, load_data

# Cấu hình trang
st.set_page_config(page_title="Dự đoán đột quỵ", page_icon="🧠", layout="centered")

# Tiêu đề trang
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🧠 Dự đoán nguy cơ đột quỵ</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Load dữ liệu và mô hình
spark = get_spark_session()
CSV_PATH = r"C:\BigData_Project\Data\Healthcare-stroke-data_after preprocess.csv"
df_pandas = load_data(CSV_PATH)
MODEL_PATH = "C:/BigData_Project/Model"
model = PipelineModel.load(MODEL_PATH)

# Hiển thị bảng dữ liệu nhỏ
with st.expander("📊 Xem dữ liệu mẫu", expanded=False):
    st.dataframe(df_pandas.head(10), use_container_width=True)

# Giao diện nhập thông tin người dùng
st.sidebar.header("📝 Nhập thông tin bệnh nhân")

user_input = {
    "hypertension": st.sidebar.radio("Tăng huyết áp (Hypertension)?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có"),
    "heart_disease": st.sidebar.radio("Bệnh tim (Heart disease)?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có"),
    "age": st.sidebar.slider("Tuổi", min_value=1, max_value=100, value=40),
    "avg_glucose_level": st.sidebar.slider("Mức đường huyết trung bình", min_value=50.0, max_value=300.0, value=100.0)
}

# Khi nhấn nút dự đoán
if st.sidebar.button("🔍 Dự đoán"):
    # Chuẩn bị dữ liệu đầu vào
    schema = StructType([
        StructField("hypertension", IntegerType(), True),
        StructField("heart_disease", IntegerType(), True),
        StructField("age", IntegerType(), True),
        StructField("avg_glucose_level", DoubleType(), True),
    ])
    user_df = spark.createDataFrame([user_input], schema=schema)

    # Dự đoán
    prediction = model.transform(user_df).collect()[0]['prediction']

    # Kết quả
    st.markdown("<hr>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            "<div style='background-color: #FFDDDD; padding: 20px; border-radius: 10px;'>"
            "<h3 style='color: red;'>💥 Nguy cơ cao bị đột quỵ!</h3>"
            "<p style='font-size:16px;'>Bạn nên đến cơ sở y tế để được tư vấn và kiểm tra kỹ lưỡng hơn.</p>"
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background-color: #D6F5D6; padding: 20px; border-radius: 10px;'>"
            "<h3 style='color: green;'>✅ Không có nguy cơ đột quỵ.</h3>"
            "<p style='font-size:16px;'>Tuy nhiên, bạn vẫn nên duy trì lối sống lành mạnh!</p>"
            "</div>", unsafe_allow_html=True)
