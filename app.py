import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import resample
import altair as alt


st.set_page_config(
    page_title="SVM Classification App",
    layout="centered"
)

st.title("Klasifikasi Data AsphaltPavementType Menggunakan Metode SVM")
st.write("Upload file CSV")

@st.cache_resource
def load_model():
    return joblib.load("model/svm_rbf.pkl")

model = load_model()

def to_500_features(row, target_len=500):
    signal = row.to_numpy(dtype=float)

    signal = signal[~np.isnan(signal)]

    if len(signal) >= target_len:
        features = signal[:target_len]
    else:
        features = np.zeros(target_len)
        features[:len(signal)] = signal

    return features

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]

        st.subheader("Review Data")
        st.dataframe(df_raw.head())

        if st.button("Klasifikasi"):
            X_features = np.vstack(
                df_raw.apply(to_500_features, axis=1)
            )

            st.success(f"Data berhasil dikonversi : {X_features.shape}")

            preds = model.predict(X_features)

            df_result = df_raw.copy()
            df_result["Predicted_Class"] = preds

            st.subheader("Hasil Klasifikasi")
            st.dataframe(df_result)

            st.subheader("Distribusi Kelas Hasil Klasifikasi")

            class_counts = (
                df_result["Predicted_Class"]
                .value_counts()
                .reset_index()
            )
            class_counts.columns = ["Class", "Count"]

            chart = (
                alt.Chart(class_counts)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Class:N", title="Kelas"),
                    y=alt.Y("Count:Q", title="Jumlah Data"),
                    tooltip=["Class", "Count"]
                )
                .properties(
                    width=600,
                    height=400
                )
            )

            st.altair_chart(chart, use_container_width=True)

            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Hasil CSV",
                csv,
                "hasil_klasifikasi.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
