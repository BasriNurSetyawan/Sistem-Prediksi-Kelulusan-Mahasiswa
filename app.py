import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Kelulusan", layout="centered")

# --- BAGIAN ATAS: INFORMASI METODE & ALGORITMA ---
st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")

st.markdown("""
### 🧠 Metode & Algoritma Sistem
* **Metode Utama:** *Supervised Machine Learning* (Klasifikasi)
* **Algoritma Model:** *Random Forest Classifier* (N-Estimators: 100)
* **Pra-pemrosesan Data:** *Standard Scaler* untuk normalisasi data waktu & *Label Encoder* untuk kategori teks.
* **Otomasi Fitur Latar Belakang:** Perhitungan otomatis metrik *Stress Level* berbasis aturan bersyarat (Logika IF-ELSE) dan *Addiction Score* menggunakan Regresi Linier *Multi-variabel*.
""")
st.divider()

# Load komponen AI
model = joblib.load('model_akademik.pkl')
scaler = joblib.load('scaler_akademik.pkl')
le_stress = joblib.load('le_stress.pkl')
le_genre = joblib.load('le_genre.pkl')
le_target = joblib.load('le_target.pkl')

# Evaluasi background
df = pd.read_csv('Gaming_Academic_Performance_updated.csv').dropna()
df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if x >= 60 else 'Tidak Lulus')

X = df[['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level']].copy()
X['stress_level'] = le_stress.transform(X['stress_level'])
X['gaming_genre'] = le_genre.transform(X['gaming_genre'])
y = le_target.transform(df['status_lulus'])

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

st.subheader("📊 Hasil Evaluasi Model")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc*100:.2f}%")
col2.metric("Precision", f"{prec*100:.2f}%")
col3.metric("Recall", f"{rec*100:.2f}%")
col4.metric("F1 Score", f"{f1*100:.2f}%")

st.write("**Grafik Hasil Prediksi (Confusion Matrix):**")
cm = confusion_matrix(y_test, y_pred)
label_0 = le_target.classes_[0]
label_1 = le_target.classes_[1]

data_cm = {
    'Kategori': [
        f'Tepat: Aktual {label_0}, Prediksi {label_0}',
        f'Meleset: Aktual {label_0}, Prediksi {label_1}',
        f'Meleset: Aktual {label_1}, Prediksi {label_0}',
        f'Tepat: Aktual {label_1}, Prediksi {label_1}'
    ],
    'Jumlah Mahasiswa': cm.flatten()
}
df_grafik_cm = pd.DataFrame(data_cm).set_index('Kategori')
st.bar_chart(df_grafik_cm)

st.divider()

# --- FORMULIR INPUT ---
st.subheader("🎮 Coba Prediksi Kelulusan")
st.write("Masukkan metrik rutinitas harian mahasiswa di bawah ini:")

with st.form("form_prediksi"):
    c1, c2 = st.columns(2)
    
    with c1:
        study = st.number_input("Lama Belajar (Jam/Hari)", min_value=0.0, max_value=24.0, value=5.0)
        gaming = st.number_input("Lama Main Game (Jam/Hari)", min_value=0.0, max_value=24.0, value=4.0)
        genre = st.selectbox("Genre Game Favorit", ['FPS', 'RPG', 'Casual'])
        attendance = st.number_input("Tingkat Kehadiran (%)", min_value=0.0, max_value=100.0, value=80.0)
        
    with c2:
        sleep = st.number_input("Lama Tidur (Jam/Hari)", min_value=0.0, max_value=24.0, value=6.5)
        device = st.number_input("Lama Pegang Gadget Total (Jam/Hari)", min_value=0.0, max_value=24.0, value=7.0)
        sosial = st.number_input("Waktu Sosialisasi/Nongkrong (Jam/Hari)", min_value=0.0, max_value=24.0, value=2.0)
        
    submit = st.form_submit_button("Jalankan Prediksi")

if submit:
    if (study + gaming + sleep + sosial) > 24:
        st.error("Total waktu belajar, nge-game, nongkrong, dan tidur tidak boleh lebih dari 24 jam.")
    elif device < gaming:
        st.error("Total waktu pegang gadget tidak boleh lebih kecil dari waktu main game.")
    else:
        # Kalkulasi background sesuai logika asli dataset
        if study > 7 and sleep <= 6:
            stress = 'High'
        elif gaming > 5:
            stress = 'Low'
        else:
            stress = 'Medium'
            
        # Rumus regresi baru setelah ditambah variabel device_usage
        addiction = -0.0024 + (1.4820 * gaming) + (0.5101 * device)
        
        st.info(f"Sistem mengkalkulasi otomatis: Skor Kecanduan = {addiction:.2f} | Tingkat Stres = {stress}")
        
        input_df = pd.DataFrame([[study, gaming, sleep, attendance, device, sosial, genre, addiction, stress]], 
                                columns=['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level'])
        
        input_df['stress_level'] = le_stress.transform(input_df['stress_level'])
        input_df['gaming_genre'] = le_genre.transform(input_df['gaming_genre'])
        input_scaled = scaler.transform(input_df)
        
        hasil_encoded = model.predict(input_scaled)
        hasil = le_target.inverse_transform(hasil_encoded)[0]
        
        if hasil == "Lulus":
            st.success(f"Hasil Prediksi AI: Mahasiswa berpotensi LULUS")
        else:
            st.error(f"Hasil Prediksi AI: Mahasiswa berpotensi TIDAK LULUS")