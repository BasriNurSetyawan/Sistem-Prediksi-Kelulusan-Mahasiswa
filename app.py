import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sistem Prediksi Kelulusan Mahasiswa", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('model_akademik.pkl')
    scaler = joblib.load('scaler_akademik.pkl')
    le_stress = joblib.load('le_stress.pkl')
    le_genre = joblib.load('le_genre.pkl')
    le_target = joblib.load('le_target.pkl')
    return model, scaler, le_stress, le_genre, le_target

model, scaler, le_stress, le_genre, le_target = load_assets()

menu = st.sidebar.selectbox("Pilih Halaman:", [
    "Beranda & Prediksi", 
    "Informasi & Metodologi", 
    "Saran Penggunaan Gadget"
])

if menu == "Beranda & Prediksi":
    st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa Berbasis AI")
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
            
            kategori_aktivitas = st.selectbox("Fokus Utama Harian", ['Akademik & Tugas', 'Gaming Kompetition', 'Seimbang / Santai'])
            
        submit = st.form_submit_button("🚀 Jalankan Prediksi AI")

    if submit:
        if (study + gaming + sleep + sosial) > 24:
            st.error("⚠️ Total waktu harian (belajar, nge-game, tidur, nongkrong) tidak boleh lebih dari 24 jam!")
        elif device < gaming:
            st.error("⚠️ Total penggunaan gadget tidak boleh lebih kecil dari durasi main game!")
        else:
            # Logika otomatisasi background
            if study > 7 and sleep <= 6:
                stress = 'High'
            elif gaming > 5:
                stress = 'Low'
            else:
                stress = 'Medium'
                
            addiction = -0.0024 + (1.4820 * gaming) + (0.5101 * device)
            
            st.info(f"🔍 Kalkulasi Otomatis Sistem — Skor Kecanduan: **{addiction:.2f}** | Tingkat Stres: **{stress}**")
            
            input_df = pd.DataFrame([[study, gaming, sleep, attendance, device, sosial, genre, addiction, stress]], 
                                    columns=['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level'])
            
            input_df['stress_level'] = le_stress.transform(input_df['stress_level'])
            input_df['gaming_genre'] = le_genre.transform(input_df['gaming_genre'])
            input_scaled = scaler.transform(input_df)
            
            hasil_encoded = model.predict(input_scaled)
            hasil = le_target.inverse_transform(hasil_encoded)[0]
            
            st.divider()
            if hasil == "Lulus":
                st.success("🎉 **Hasil Prediksi:** Mahasiswa ini memiliki probabilitas tinggi untuk **LULUS** tepat waktu.")
            else:
                st.error("🚨 **Hasil Prediksi:** Mahasiswa ini berisiko **TIDAK LULUS**. Perlu evaluasi manajemen waktu.")

elif menu == "Informasi & Metodologi":
    st.title("📚 Informasi Sistem & Metodologi Penelitian")
    st.write("Halaman ini merangkum arsitektur di balik layar, algoritma yang digunakan, serta evaluasi performa model.")
    
    st.subheader("Arsitektur Algoritma")
    st.info("""
    * **Algoritma Klasifikasi:** Menggunakan *Random Forest Classifier* dengan total 100 *Decision Trees* (`n_estimators=100`).
    * **Pra-pemrosesan Data:** 
      * `StandardScaler` untuk normalisasi rentang angka (durasi jam, kehadiran).
      * `LabelEncoder` untuk mengubah data kategori teks (Genre Game, Tingkat Stres) menjadi numerik.
    * **Variabel Penentu Utama:** Berdasarkan uji *Feature Importance*, jam belajar memegang pengaruh terbesar (sekitar 43%), disusul oleh durasi gaming dan kualitas tidur.
    """)
    
    st.subheader("Evaluasi Performa Model (Testing Data)")
    
    # Load data evaluasi untuk ditampilkan metriknya
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
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi Model", f"{acc*100:.2f}%")
    m2.metric("Precision Score", f"{prec*100:.2f}%")
    m3.metric("Recall Score", f"{rec*100:.2f}%")
    
    st.write("**Visualisasi Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    label_0, label_1 = le_target.classes_[0], le_target.classes_[1]
    df_cm = pd.DataFrame({
        'Kategori Evaluasi': [
            f'Aktual {label_0} (Prediksi {label_0})', f'Aktual {label_0} (Prediksi {label_1})',
            f'Aktual {label_1} (Prediksi {label_0})', f'Aktual {label_1} (Prediksi {label_1})'
        ],
        'Jumlah Sampel': cm.flatten()
    }).set_index('Kategori Evaluasi')
    st.bar_chart(df_cm)

elif menu == "Saran Penggunaan Gadget":
    st.title("💡 Panduan & Saran Manajemen Gadget Bagi Mahasiswa")
    st.write("Rekomendasi taktis berbasis analisis data untuk menyeimbangkan hobi gaming, penggunaan gadget, dan performa akademik.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🛑 Risiko Durasi Gadget Berlebih")
        st.warning("""
        * **Penurunan Jam Tidur:** Gadget yang digunakan larut malam mengganggu produksi melatonin, berakibat pada penurunan fokus kuliah keesokan harinya.
        * **Lonjakan Skor Kecanduan:** Berdasarkan rumus regresi sistem, setiap penambahan durasi gaming dan total layar secara linear mendongkrak tingkat risiko adiksi.
        * **Distraksi Akademik:** Waktu belajar yang terpotong untuk aktivitas digital sekunder berbanding lurus dengan penurunan indeks prestasi.
        """)
        
    with col_b:
        st.subheader("✅ Tips & Solusi Bijak")
        st.success("""
        * **Terapkan Batasan Waktu (Time Boxing):** Alokasikan waktu maksimal gaming (misal: 2 jam sehari setelah tugas beres).
        * **Prioritaskan Porsi Belajar:** Pastikan rasio jam belajar minimal 2 kali lipat dari durasi hiburan harian.
        * **Jeda Istirahat Layar:** Gunakan metode *20-20-20* (tiap 20 menit menatap layar, istirahatkan mata sejenak melihat objek sejauh 20 kaki) untuk menjaga kesehatan mata.
        """)
        
    st.info("📌 **Kesimpulan:** Main game atau menggunakan gadget tidak dilarang, asalkan diimbangi dengan tingkat kehadiran yang tinggi dan waktu istirahat yang cukup agar target kelulusan tetap tercapai.")
