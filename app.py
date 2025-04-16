import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Deteksi Kanker Paru-paru",
    page_icon="🫁",
    layout="wide"
)

# Tampilan judul
st.title("🫁 Sistem Deteksi Kanker Paru-paru")
st.markdown("#### Aplikasi untuk mendeteksi risiko kanker paru-paru berdasarkan faktor risiko pasien")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        # Ganti dengan path model Anda
        with open('./cancerModel100.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        st.error("Model tidak ditemukan. Silakan upload model terlebih dahulu.")
        return None

# Fungsi untuk memuat contoh dataset
@st.cache_data
def load_sample_data():
    # Contoh data dari input pengguna
    data = {
        "index": [0, 1, 2],
        "Patient Id": ["P1", "P10", "P100"],
        "Age": [33, 17, 35],
        "Gender": [1, 1, 1],
        "Air Pollution": [2, 3, 4],
        "Alcohol use": [4, 1, 5],
        "Dust Allergy": [5, 5, 6],
        "OccuPational Hazards": [4, 3, 5],
        "Genetic Risk": [3, 4, 5],
        "chronic Lung Disease": [2, 2, 4],
        "Balanced Diet": [2, 2, 6],
        "Obesity": [4, 2, 7],
        "Smoking": [3, 2, 2],
        "Passive Smoker": [2, 4, 3],
        "Chest Pain": [2, 2, 4],
        "Coughing of Blood": [4, 3, 8],
        "Fatigue": [3, 1, 8],
        "Weight Loss": [4, 3, 7],
        "Shortness of Breath": [2, 7, 9],
        "Wheezing": [2, 8, 2],
        "Swallowing Difficulty": [3, 6, 1],
        "Clubbing of Finger Nails": [1, 2, 4],
        "Frequent Cold": [2, 1, 6],
        "Dry Cough": [3, 7, 7],
        "Snoring": [4, 2, 2],
        "Level": ["Low", "Medium", "High"]
    }
    return pd.DataFrame(data)

# Memuat model dan contoh data
model = load_model()
sample_data = load_sample_data()

# Sidebar untuk opsi
st.sidebar.header("Upload dan Konfigurasi")

# Opsi untuk mengupload model
uploaded_model = st.sidebar.file_uploader("Upload Model (pickle file)", type=["pkl"])
if uploaded_model is not None:
    with open("temp_model.pkl", "wb") as f:
        f.write(uploaded_model.getbuffer())
    model = pickle.load(open("temp_model.pkl", "rb"))
    st.sidebar.success("Model berhasil dimuat!")

# Contoh data
st.sidebar.header("Dataset Contoh")
st.sidebar.dataframe(sample_data.head(3), use_container_width=True)

# Tampilkan tab
tab1, tab2, tab3 = st.tabs(["Prediksi Individual", "Upload Dataset", "Tentang"])

with tab1:
    st.header("Masukkan Data Pasien")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_id = st.text_input("ID Pasien", "P101")
        age = st.number_input("Usia", min_value=1, max_value=100, value=30)
        gender = st.selectbox("Jenis Kelamin", options=[1, 2], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
        air_pollution = st.slider("Polusi Udara", 1, 10, 3)
        alcohol_use = st.slider("Penggunaan Alkohol", 1, 10, 3)
        dust_allergy = st.slider("Alergi Debu", 1, 10, 3)
        occupational_hazards = st.slider("Bahaya Pekerjaan", 1, 10, 3)
        genetic_risk = st.slider("Risiko Genetik", 1, 10, 3)
    
    with col2:
        chronic_lung_disease = st.slider("Penyakit Paru Kronis", 1, 10, 3)
        balanced_diet = st.slider("Diet Seimbang", 1, 10, 3)
        obesity = st.slider("Obesitas", 1, 10, 3)
        smoking = st.slider("Merokok", 1, 10, 3)
        passive_smoker = st.slider("Perokok Pasif", 1, 10, 3)
        chest_pain = st.slider("Nyeri Dada", 1, 10, 3)
        coughing_blood = st.slider("Batuk Darah", 1, 10, 3)
        fatigue = st.slider("Kelelahan", 1, 10, 3)

    with col3:
        weight_loss = st.slider("Penurunan Berat Badan", 1, 10, 3)
        shortness_breath = st.slider("Sesak Napas", 1, 10, 3)
        wheezing = st.slider("Mengi", 1, 10, 3)
        swallowing_difficulty = st.slider("Kesulitan Menelan", 1, 10, 3)
        clubbing_finger_nails = st.slider("Jari Tabuh", 1, 10, 3)
        frequent_cold = st.slider("Sering Pilek", 1, 10, 3)
        dry_cough = st.slider("Batuk Kering", 1, 10, 3)
        snoring = st.slider("Mendengkur", 1, 10, 3)
    
    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Risiko Kanker"):
        if model:
            # Persiapan data untuk prediksi
            input_data = [[
                age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
                genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking,
                passive_smoker, chest_pain, coughing_blood, fatigue, weight_loss,
                shortness_breath, wheezing, swallowing_difficulty, clubbing_finger_nails,
                frequent_cold, dry_cough, snoring
            ]]
            
            try:
                # Prediksi
                prediction = model.predict(input_data)
                predict_proba = model.predict_proba(input_data)
                
                # Tampilkan hasil prediksi
                risk_level = prediction[0]
                st.subheader("Hasil Prediksi:")
                
                if risk_level == "Low":
                    st.success(f"Risiko Kanker Paru-paru: RENDAH")
                    risk_icon = "✅"
                elif risk_level == "Medium":
                    st.warning(f"Risiko Kanker Paru-paru: SEDANG")
                    risk_icon = "⚠️"
                else:
                    st.error(f"Risiko Kanker Paru-paru: TINGGI")
                    risk_icon = "🚨"
                
                # Menampilkan probabilitas
                st.subheader("Probabilitas Setiap Kelas:")
                prob_df = pd.DataFrame({
                    'Kelas': model.classes_,
                    'Probabilitas': predict_proba[0]
                })
                
                # Menampilkan chart
                st.bar_chart(prob_df.set_index('Kelas'))
                
                # Menampilkan rekomendasi
                st.subheader("Rekomendasi:")
                if risk_level == "Low":
                    st.write("- Tetap menjaga gaya hidup sehat")
                    st.write("- Pemeriksaan rutin tahunan disarankan")
                    st.write("- Hindari paparan asap rokok dan polusi")
                elif risk_level == "Medium":
                    st.write("- Konsultasi dengan dokter spesialis paru")
                    st.write("- Evaluasi faktor risiko yang dapat dimodifikasi")
                    st.write("- Pemeriksaan lanjutan mungkin diperlukan")
                else:
                    st.write("- Segera konsultasi dengan dokter spesialis paru-paru")
                    st.write("- Pemeriksaan diagnostik lanjut sangat disarankan")
                    st.write("- Evaluasi dan penanganan faktor risiko mendesak")
                
                # Simpan hasil prediksi
                st.session_state.last_prediction = {
                    'Patient ID': patient_id,
                    'Age': age,
                    'Risk Level': risk_level,
                    'Probabilities': predict_proba[0]
                }
                
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam prediksi: {e}")
        else:
            st.error("Model belum dimuat. Silakan upload model terlebih dahulu.")

with tab2:
    st.header("Prediksi dari Dataset")
    
    uploaded_file = st.file_uploader("Upload file dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data yang diupload:")
            st.dataframe(data.head(), use_container_width=True)
            
            if st.button("Prediksi Semua Data"):
                if model:
                    # Memisahkan fitur dan target jika ada
                    if "Level" in data.columns:
                        features = data.drop(["index", "Patient Id", "Level"], axis=1, errors='ignore')
                    else:
                        features = data.drop(["index", "Patient Id"], axis=1, errors='ignore')
                    
                    # Prediksi
                    predictions = model.predict(features)
                    probabilities = model.predict_proba(features)
                    
                    # Gabungkan dengan data asli
                    result_df = data.copy()
                    result_df['Predicted_Level'] = predictions
                    
                    # Tambahkan probabilitas
                    for idx, cls in enumerate(model.classes_):
                        result_df[f'Prob_{cls}'] = probabilities[:, idx]
                    
                    st.subheader("Hasil Prediksi:")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Distribusi hasil prediksi
                    st.subheader("Distribusi Prediksi:")
                    pred_counts = result_df['Predicted_Level'].value_counts().reset_index()
                    pred_counts.columns = ['Level', 'Count']
                    st.bar_chart(pred_counts.set_index('Level'))
                    
                    # Download hasil
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Hasil Prediksi (CSV)",
                        data=csv,
                        file_name="hasil_prediksi_kanker_paru.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Model belum dimuat. Silakan upload model terlebih dahulu.")
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam memproses file: {e}")

with tab3:
    st.header("Tentang Aplikasi")
    
    st.markdown("""
    ### Deteksi Kanker Paru-paru
    
    Aplikasi ini dirancang untuk membantu tenaga medis dalam melakukan deteksi awal risiko kanker paru-paru berdasarkan berbagai faktor risiko dan gejala pasien.
    
    #### Dataset
    Dataset yang digunakan mencakup beberapa variabel penting:
    - **Informasi Demografi**: Usia, Jenis Kelamin
    - **Faktor Lingkungan**: Polusi Udara, Alergi Debu, Bahaya Pekerjaan
    - **Faktor Gaya Hidup**: Penggunaan Alkohol, Diet Seimbang, Obesitas, Merokok
    - **Faktor Genetik**: Risiko Genetik, Penyakit Paru Kronis
    - **Gejala**: Nyeri Dada, Batuk Darah, Kelelahan, Penurunan Berat Badan, dll.
    
    #### Model
    Aplikasi menggunakan model machine learning yang telah dilatih untuk mengklasifikasikan risiko kanker paru-paru menjadi tiga tingkat:
    - **Rendah (Low)**: Risiko rendah terkena kanker paru-paru
    - **Sedang (Medium)**: Risiko sedang, pemeriksaan lanjutan mungkin diperlukan
    - **Tinggi (High)**: Risiko tinggi, perlu perhatian medis segera
    
    #### Catatan Penting
    Aplikasi ini hanya sebagai alat bantu dan tidak menggantikan diagnosis medis profesional. Hasil prediksi sebaiknya didiskusikan dengan dokter untuk evaluasi lebih lanjut.
    """)
    
    st.subheader("Cara Penggunaan")
    
    st.markdown("""
    1. **Prediksi Individual**:
       - Masukkan data pasien pada form yang tersedia
       - Klik tombol "Prediksi Risiko Kanker" untuk melihat hasil
       
    2. **Prediksi Massal**:
       - Upload file CSV dengan format yang sesuai
       - Klik tombol "Prediksi Semua Data" untuk memprediksi seluruh dataset
       - Download hasil prediksi untuk analisis lebih lanjut
       
    3. **Upload Model**:
       - Jika memiliki model custom, upload file pickle model di sidebar
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Copyright Wahyu Andika Rahadi | Sistem Deteksi Kanker Paru-paru | Dibuat untuk keperluan medis dan penelitian")