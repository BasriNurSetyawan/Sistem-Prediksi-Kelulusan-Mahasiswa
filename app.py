import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Sistem Prediksi Akademik", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    model = joblib.load('model_akademik.pkl')
    scaler = joblib.load('scaler_akademik.pkl')
    le_stress = joblib.load('le_stress.pkl')
    le_target = joblib.load('le_target.pkl')
    return model, scaler, le_stress, le_target

@st.cache_data
def load_data():
    df = pd.read_csv('Gaming_Academic_Performance_updated.csv').dropna()
    if 'assignment_completion' not in df.columns:
        df['assignment_completion'] = df['grades'].apply(lambda x: min(100.0, max(30.0, float(x) + 5.0)) if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit() else 75.0)
    
    if 'grades' in df.columns and df['grades'].dtype == 'object':
        df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if str(x).lower() in ['lulus', 'pass', 'b', 'a'] else 'Tidak Lulus')
    else:
        df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if float(x) >= 60 else 'Tidak Lulus')
    return df

try:
    model, scaler, le_stress, le_target = load_models()
    df = load_data()
except Exception as e:
    st.error("Error: Pastikan telah menjalankan file `train.py` terlebih dahulu untuk men-generate file .pkl")
    st.stop()

st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio("Pilih Menu:", [
    "🏠 Home", 
    "📊 Dataset", 
    "⚙️ Preprocessing", 
    "🌲 Training Model", 
    "📈 Evaluasi", 
    "🎯 Prediksi", 
])


if menu == "🏠 Home":
    st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Deskripsi")
        st.write("Sistem cerdas berbasis *Machine Learning* untuk memprediksi probabilitas kelulusan mahasiswa berdasarkan kebiasaan harian seperti waktu belajar, bermain game, penggunaan gadget, dan kehadiran kuliah.")
        
        st.write("Tujuan")
        st.info("Membantu mahasiswa mengidentifikasi risiko akademis sejak dini dan memberikan rekomendasi otomatis untuk manajemen waktu yang lebih baik.")
    with col2:
        st.success("Cara Kerja Sistem\n"
                   "1. **Input Data**: Pengguna memasukkan data harian.\n"
                   "2. **Proses AI**: Data dianalisis menggunakan algoritma Random Forest.\n"
                   "3. **Prediksi**: Output berupa probabilitas lulus/tidak lulus.\n"
                   "4. **Rekomendasi**: Sistem memberikan saran korektif.")
        
    st.write("Pipeline Machine Learning")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Machine_learning_pipeline.svg/1024px-Machine_learning_pipeline.svg.png", use_container_width=True)
    st.caption("Ilustrasi Alur Machine Learning standar.")


elif menu == "📊 Dataset":
    st.title("📊 Eksplorasi Dataset")
    st.write("Dataset yang digunakan berfokus pada keseimbangan aktivitas akademik dan hiburan (gaming) mahasiswa.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())
    
    st.subheader("Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander("Lihat Tipe Data Fitur"):
        st.write(pd.DataFrame(df.dtypes, columns=["Tipe Data"]))
        
    with st.expander("Lihat Statistik Dataset (Describe)"):
        st.write(df.describe())

    st.subheader("Distribusi Target Kelulusan")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='status_lulus', data=df, palette='viridis', ax=ax)
    plt.title("Distribusi Kelas Lulus vs Tidak Lulus")
    st.pyplot(fig)


elif menu == "⚙️ Preprocessing":
    st.title("⚙️ Tahapan Preprocessing Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning", "Encoding", "Scaling", "Train/Test Split"])
    
    with tab1:
        st.subheader("Data Cleaning")
        st.write("Membersihkan dataset dari *missing values* (Null/NaN) dengan melakukan `dropna()`. Selain itu, kami melakukan konversi target nilai numerik menjadi label 'Lulus' (>=60) dan 'Tidak Lulus' (<60).")
    
    with tab2:
        st.subheader("Label Encoding")
        st.write("Mengubah data kategori teks menjadi angka yang bisa dipahami mesin menggunakan `LabelEncoder` dari Scikit-Learn.")
        st.code("X['stress_level'] = le_stress.fit_transform(X['stress_level'])\ny_encoded = le_target.fit_transform(y)", language='python')
        
    with tab3:
        st.subheader("Standardization (Scaling)")
        st.write("Menggunakan `StandardScaler` untuk menyamakan skala (range) data angka seperti jam belajar dan persentase kehadiran agar tidak ada fitur yang mendominasi algoritma.")
        st.code("scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)", language='python')
        
    with tab4:
        st.subheader("Pemisahan Data (Train/Test Split)")
        st.write("Membagi dataset menjadi 80% data latih (Train) dan 20% data uji (Test). Kami menggunakan parameter `stratify=y` untuk menjaga keseimbangan rasio kelas target pada proses pembagian data.")
        st.code("X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)", language='python')


elif menu == "🌲 Training Model":
    st.title("🌲 Training Model: Random Forest")
    
    with st.expander("Apa itu Random Forest & Cara Kerjanya?", expanded=True):
        st.write("""
        **Random Forest** adalah algoritma *ensemble learning* yang terdiri dari banyak Decision Tree. 
        Cara kerjanya adalah dengan membangun banyak 'pohon keputusan' saat masa training, lalu hasil akhir ditentukan dari voting terbanyak (untuk klasifikasi) dari seluruh pohon tersebut.
        """)
        
    st.subheader("Alasan Pemilihan Algoritma")
    st.success("- Tahan terhadap kondisi *overfitting*.\n- Mampu menangani fitur yang tidak linear.\n- Secara bawaan bisa mengukur tingkat kepentingan setiap fitur (Feature Importance).")
    
    st.subheader("Parameter Model yang Digunakan")
    col1, col2 = st.columns(2)
    col1.info("- **n_estimators**: 200\n- **criterion**: gini\n- **max_depth**: 10")
    col2.info("- **min_samples_split**: 2\n- **min_samples_leaf**: 1\n- **random_state**: 42")
    
    st.subheader("Ringkasan Training")
    st.write("Model dilatih menggunakan dataset yang telah melalui tahapan preprocessing, dengan arsitektur 200 pohon berkedalaman maksimal 10 level untuk menjaga performa tanpa kehilangan akurasi.")


elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi Performa Model")
    
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
        c2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
        c3.metric("Recall", f"{metrics['Recall']*100:.2f}%")
        c4.metric("F1 Score", f"{metrics['F1_Score']*100:.2f}%")
        
        
        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.subheader("Confusion Matrix")
            cm_data = np.array(metrics['Confusion_Matrix'])
            fig1, ax1 = plt.subplots(figsize=(5,4))
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
            plt.ylabel('Aktual')
            plt.xlabel('Prediksi')
            st.pyplot(fig1)
            
        with col_fi:
            st.subheader("Feature Importance")
            if os.path.exists('feature_importance.csv'):
                fi_df = pd.read_csv('feature_importance.csv')
                fig2, ax2 = plt.subplots(figsize=(5,4))
                sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
                st.pyplot(fig2)
            else:
                st.write("File feature_importance.csv tidak ditemukan.")
                
        st.subheader("Korelasi Fitur (Heatmap)")
        df_num = df.copy()
        df_num['stress_level'] = le_stress.transform(df_num['stress_level'])
        df_num['status_lulus'] = le_target.transform(df_num['status_lulus'])
        corr = df_num[['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'assignment_completion', 'device_usage', 'status_lulus']].corr()
        fig3, ax3 = plt.subplots(figsize=(8,4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig3)
        
        with st.expander("Lihat Detail Classification Report"):
            st.json(metrics['Classification_Report'])
    else:
        st.error("File metrics.json tidak ditemukan. Silakan jalankan train.py terlebih dahulu.")


elif menu == "🎯 Prediksi":
    st.title("🎯 Form Prediksi Kelulusan")
    st.write("Masukkan kebiasaan harian mahasiswa pada form di bawah ini.")
    
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        with c1:
            study = st.number_input("Lama Belajar (Jam/Hari)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
            gaming = st.number_input("Lama Main Game (Jam/Hari)", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
            assignment = st.number_input("Tingkat Kepatuhan Tugas (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            attendance = st.number_input("Kehadiran Kuliah (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
            
        with c2:
            sleep = st.number_input("Lama Tidur (Jam/Hari)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            device = st.number_input("Penggunaan Gadget (Jam/Hari)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            sosial = st.number_input("Waktu Sosialisasi (Jam/Hari)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
            
        submit = st.form_submit_button("Jalankan Prediksi AI")

    if submit:
        if (study + gaming + sleep + sosial) > 24:
            st.error("Total waktu harian (belajar, main, tidur, sosial) tidak boleh melebihi 24 jam!")
        elif device < gaming:
            st.error("Total penggunaan gadget tidak boleh lebih kecil dari durasi main game!")
        else:
            if study > 7 and sleep <= 6:
                stress = 'High'
            elif gaming > 5:
                stress = 'Low'
            else:
                stress = 'Medium'
            addiction = -0.0024 + (1.4820 * gaming) + (0.5101 * device)
            
            input_data = pd.DataFrame([[study, gaming, sleep, attendance, assignment, device, sosial, addiction, stress]], 
                                      columns=['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'assignment_completion', 'device_usage', 'social_activity', 'addiction_score', 'stress_level'])
            input_data['stress_level'] = le_stress.transform(input_data['stress_level'])
            input_scaled = scaler.transform(input_data)
            
            hasil_encoded = model.predict(input_scaled)
            hasil = le_target.inverse_transform(hasil_encoded)[0]
            probabilitas = model.predict_proba(input_scaled)[0]
            
            kelas_labels = le_target.classes_
            prob_dict = {kelas_labels[i]: probabilitas[i]*100 for i in range(len(kelas_labels))}
            
            st.divider()
            
            tab_hasil, tab_saran = st.tabs(["Hasil Prediksi & Probabilitas", "Simulasi Perbaikan Otomatis"])
            
            with tab_hasil:
                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.subheader("Status Prediksi")
                    if hasil == "Lulus":
                        st.success("✅ Prediksi: Mahasiswa berpotensi besar LULUS")
                        st.write("**Penjelasan:** Pola kebiasaan harian yang diinput memiliki kemiripan historis yang tinggi dengan data mahasiswa yang berhasil lulus.")
                    else:
                        st.error("❌ Prediksi: Mahasiswa berisiko TIDAK LULUS")
                        st.write("**Penjelasan:** Terdapat kebiasaan ekstrem yang terdeteksi (seperti kurang tidur atau main game berlebih) yang membahayakan performa akademik.")
                
                with res_c2:
                    st.subheader("Detail Predict Probability")
                    st.metric("Probabilitas Lulus", f"{prob_dict.get('Lulus', 0):.2f} %")
                    st.metric("Probabilitas Tidak Lulus", f"{prob_dict.get('Tidak Lulus', 0):.2f} %")
            
            with tab_saran:
                st.write("Rekomendasi taktis untuk menyeimbangkan pola hidup (berdasarkan batas standar bisnis):")
                rekomendasi = []
                
                if study < 2.0: rekomendasi.append(f"Tambah jam belajar minimal menjadi **2.0 jam** (+{2.0 - study:.1f} jam).")
                if sleep < 8.0: rekomendasi.append(f"Tingkatkan tidur minimal menjadi **8.0 jam** (+{8.0 - sleep:.1f} jam).")
                if gaming > 6.0: rekomendasi.append(f"Kurangi durasi game maksimal menjadi **6.0 jam** (-{gaming - 6.0:.1f} jam).")
                if device > 10.0: rekomendasi.append(f"Batasi penggunaan gadget maksimal menjadi **10.0 jam**.")
                if attendance < 75.0: rekomendasi.append("Tingkatkan kehadiran kuliah minimal **75%**.")
                if assignment < 75.0: rekomendasi.append("Tingkatkan kepatuhan tugas minimal **75%**.")
                
                if rekomendasi:
                    for idx, rec in enumerate(rekomendasi[:4]):
                        st.warning(f"{idx+1}. {rec}")
                else:
                    st.info("Seluruh kebiasaan harian Anda sudah berada dalam batas ideal yang disarankan.")