import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sistem Prediksi Kelulusan Mahasiswa", layout="wide")

# --- LOAD KOMPONEN AI ---
@st.cache_resource
def load_assets():
    model = joblib.load('model_akademik.pkl')
    scaler = joblib.load('scaler_akademik.pkl')
    le_stress = joblib.load('le_stress.pkl')
    le_genre = joblib.load('le_genre.pkl')
    le_target = joblib.load('le_target.pkl')
    return model, scaler, le_stress, le_genre, le_target

model, scaler, le_stress, le_genre, le_target = load_assets()

# --- NAVBAR DI KIRI ATAS (SIDEBAR) ---
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman:", [
    "Beranda & Prediksi", 
    "Informasi & Metodologi", 
    "Saran Penggunaan Gadget"
])

# ==========================================
# HALAMAN 1: BERANDA & PREDIKSI UTAMA
# ==========================================
if menu == "Beranda & Prediksi":
    st.title("Beranda dan Prediksi Kelulusan Mahasiswa")
    st.write("Silakan masukkan data kebiasaan harian mahasiswa pada formulir di bawah ini untuk melihat estimasi status kelulusannya.")
    
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        
        with c1:
            study = st.number_input("Lama Belajar (Jam/Hari)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
            gaming = st.number_input("Lama Main Game (Jam/Hari)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
            genre = st.selectbox("Genre Game Favorit", ['FPS', 'RPG', 'Casual'])
            attendance = st.number_input("Tingkat Kehadiran Kuliah (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
            
        with c2:
            sleep = st.number_input("Lama Tidur (Jam/Hari)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            device = st.number_input("Total Penggunaan Gadget (Jam/Hari)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            sosial = st.number_input("Waktu Sosialisasi / Nongkrong (Jam/Hari)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
            
        submit = st.form_submit_button("Jalankan Prediksi AI")

    if submit:
        if (study + gaming + sleep + sosial) > 24:
            st.error("Total waktu harian (belajar, nge-game, tidur, nongkrong) tidak boleh lebih dari 24 jam!")
        elif device < gaming:
            st.error("Total penggunaan gadget tidak boleh lebih kecil dari durasi main game!")
        else:
            # Kalkulasi otomatis background
            if study > 7 and sleep <= 6:
                stress = 'High'
            elif gaming > 5:
                stress = 'Low'
            else:
                stress = 'Medium'
                
            addiction = -0.0024 + (1.4820 * gaming) + (0.5101 * device)
            
            st.info(f"Kalkulasi Otomatis Sistem — Skor Kecanduan: {addiction:.2f} | Tingkat Stres: {stress}")
            
            input_df = pd.DataFrame([[study, gaming, sleep, attendance, device, sosial, genre, addiction, stress]], 
                                    columns=['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level'])
            
            input_df['stress_level'] = le_stress.transform(input_df['stress_level'])
            input_df['gaming_genre'] = le_genre.transform(input_df['gaming_genre'])
            input_scaled = scaler.transform(input_df)
            
            hasil_encoded = model.predict(input_scaled)
            hasil = le_target.inverse_transform(hasil_encoded)[0]
            
            st.divider()
            
            # TAMPILAN HASIL UTAMA
            if hasil == "Lulus":
                st.success("Hasil Prediksi: Mahasiswa ini memiliki probabilitas tinggi untuk LULUS tepat waktu.")
            else:
                st.error("Hasil Prediksi: Mahasiswa ini berisiko TIDAK LULUS.")
                
                # --- SIMULASI PERBAIKAN OTOMATIS BERDASARKAN KETENTUAN BARU ---
                st.subheader("Simulasi Perbaikan Otomatis")
                st.write("Berikut adalah rekomendasi penyesuaian berdasarkan batas standar (Belajar min. 2 jam, Tidur min. 8 jam, Main Game maks. 6 jam, Gadget maks. 10 jam):")
                
                sim_study = study
                sim_gaming = gaming
                sim_device = device
                sim_sleep = sleep
                sim_attendance = attendance
                sim_hasil = hasil
                
                rekomendasi_list = []
                
                # Cek dan sesuaikan berdasarkan ketentuan user
                if sim_study < 2.0:
                    selisih = 2.0 - sim_study
                    sim_study = 2.0
                    rekomendasi_list.append(f"- Tambah jam belajar minimal menjadi **2 jam/hari** (butuh +{selisih} jam)")
                
                if sim_sleep < 8.0:
                    selisih = 8.0 - sim_sleep
                    sim_sleep = 8.0
                    rekomendasi_list.append(f"- Tingkatkan waktu tidur minimal menjadi **8 jam/hari** (butuh +{selisih} jam)")
                
                if sim_gaming > 6.0:
                    selisih = sim_gaming - 6.0
                    sim_gaming = 6.0
                    rekomendasi_list.append(f"- Kurangi durasi main game maksimal menjadi **6 jam/hari** (turun -{selisih} jam)")
                
                if sim_device > 10.0:
                    selisih = sim_device - 10.0
                    sim_device = 10.0
                    rekomendasi_list.append(f"- Batasi penggunaan gadget maksimal menjadi **10 jam/hari** (turun -{selisih} jam)")
                
                if sim_attendance < 75.0:
                    sim_attendance = 75.0
                    rekomendasi_list.append("- Tingkatkan kehadiran kuliah minimal menjadi **75%**")

                # Validasi ulang dengan model AI setelah penyesuaian aturan
                if sim_study > 7 and sim_sleep <= 6:
                    sim_stress = 'High'
                elif sim_gaming > 5:
                    sim_stress = 'Low'
                else:
                    sim_stress = 'Medium'
                    
                sim_addiction = -0.0024 + (1.4820 * sim_gaming) + (0.5101 * sim_device)
                sim_df = pd.DataFrame([[sim_study, sim_gaming, sim_sleep, sim_attendance, sim_device, sosial, genre, sim_addiction, sim_stress]], 
                                        columns=['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level'])
                sim_df['stress_level'] = le_stress.transform(sim_df['stress_level'])
                sim_df['gaming_genre'] = le_genre.transform(sim_df['gaming_genre'])
                sim_scaled = scaler.transform(sim_df)
                sim_hasil = le_target.inverse_transform(model.predict(sim_scaled))[0]

                if rekomendasi_list:
                    teks_rekomendasi = "Rekomendasi penyesuaian kebiasaan:\n" + "\n".join(rekomendasi_list)
                    st.warning(teks_rekomendasi)
                else:
                    st.info("Kebiasaan harian sudah sesuai batas ketentuan, namun kombinasi variabel lain masih mempengaruhi hasil prediksinya.")

# ==========================================
# HALAMAN 2: INFORMASI & METODOLOGI
# ==========================================
elif menu == "Informasi & Metodologi":
    st.title("Informasi Sistem dan Metodologi Penelitian")
    st.write("Halaman ini merangkum arsitektur di balik layar, algoritma yang digunakan, serta evaluasi performa model.")
    
    st.markdown("---")
    st.subheader("Arsitektur Algoritma")
    st.info("""
    - Algoritma Klasifikasi: Random Forest Classifier (n_estimators=100).
    - Pra-pemrosesan Data: StandardScaler untuk normalisasi angka dan LabelEncoder untuk kategori teks.
    - Variabel Penentu: Berdasarkan Feature Importance, jam belajar dan kehadiran memegang pengaruh terbesar.
    """)
    
    st.subheader("Evaluasi Performa Model")
    df = pd.read_csv('Gaming_Academic_Performance_updated.csv').dropna()
    df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if x >= 60 else 'Tidak Lulus')
    X = df[['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level']].copy()
    X['stress_level'] = le_stress.transform(X['stress_level'])
    X['gaming_genre'] = le_genre.transform(X['gaming_genre'])
    y = le_target.transform(df['status_lulus'])
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    m1, m2 = st.columns(2)
    m1.metric("Akurasi Model", f"{acc*100:.2f}%")
    m2.metric("Precision Score", f"{prec*100:.2f}%")
    
    st.write("Visualisasi Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame({'Jumlah Sampel': cm.flatten()}).set_index(pd.Index(['True Neg', 'False Pos', 'False Neg', 'True Pos']))
    st.bar_chart(df_cm)

# ==========================================
# HALAMAN 3: SARAN PENGGUNAAN GADGET
# ==========================================
elif menu == "Saran Penggunaan Gadget":
    st.title("Panduan dan Saran Manajemen Gadget Bagi Mahasiswa")
    st.write("Rekomendasi taktis berbasis analisis data untuk menyeimbangkan hobi gaming dan performa akademik.")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risiko Durasi Gadget Berlebih")
        st.write("Penggunaan gadget dan gaming tanpa kontrol berpotensi menurunkan jam tidur, mengganggu fokus belajar, serta meningkatkan skor kecanduan.")
    with c2:
        st.subheader("Tips dan Solusi Bijak")
        st.write("Terapkan manajemen waktu harian, utamakan jam belajar minimal dua kali lipat dari durasi hiburan, dan jaga waktu istirahat yang cukup.")
