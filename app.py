import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Kelulusan", layout="centered")

st.title("Sistem Prediksi Kelulusan Mahasiswa")

st.markdown("""
### Metode & Algoritma Sistem
* **Metode Utama:** *Supervised Machine Learning* (Klasifikasi)
* **Algoritma Model:** *Random Forest Classifier* (N-Estimators: 100)
* **Pra-pemrosesan Data:** *Standard Scaler* untuk normalisasi data waktu & *Label Encoder* untuk kategori teks.
* **Otomasi Fitur Latar Belakang:** Perhitungan otomatis metrik *Stress Level* berbasis aturan bersyarat (Logika IF-ELSE) dan *Addiction Score* menggunakan Regresi Linier *Multi-variabel*.
""")
st.divider()

model = joblib.load('model_akademik.pkl')
scaler = joblib.load('scaler_akademik.pkl')
le_stress = joblib.load('le_stress.pkl')
le_genre = joblib.load('le_genre.pkl')
le_target = joblib.load('le_target.pkl')

df = pd.read_csv('Gaming_Academic_Performance_updated.csv').dropna()
df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if x >= 60 else 'Tidak Lulus')

X = df[['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level']].copy()
X['stress_level'] = le_stress.transform(X['stress_level'])
X['gaming_genre'] = le_genre.transform(X['gaming_genre'])
y = le_target.transform(df['status_lulus'])

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

st.subheader("Hasil Evaluasi Model")
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

st.subheader("Coba Prediksi Kelulusan")
st.write("Masukkan metrik rutinitas harian mahasiswa di bawah ini:")

st.subheader("Prediksi Kelulusan Mahasiswa")

with st.form("form_prediksi"):

    study = st.text_input("Lama Belajar (Jam/Hari)", "5")
    gaming = st.text_input("Lama Main Game (Jam/Hari)", "4")
    sleep = st.text_input("Lama Tidur (Jam/Hari)", "6")
    attendance = st.text_input("Kehadiran (%)", "80")
    device = st.text_input("Penggunaan Gadget (Jam/Hari)", "7")
    sosial = st.text_input("Waktu Sosialisasi (Jam/Hari)", "2")
    genre = st.text_input("Genre Game (FPS/RPG/Casual)", "FPS")

    submit = st.form_submit_button("Prediksi")

if submit:

    try:
        study = float(study)
        gaming = float(gaming)
        sleep = float(sleep)
        attendance = float(attendance)
        device = float(device)
        sosial = float(sosial)
    except ValueError:
        st.error("Semua input angka harus berupa angka.")
        st.stop()

    genre = genre.strip().title()

    if genre not in ["FPS", "RPG", "Casual"]:
        st.error("Genre hanya boleh FPS, RPG, atau Casual.")
        st.stop()

    if study < 0 or study > 24:
        st.error("Jam belajar harus antara 0-24.")
        st.stop()

    if gaming < 0 or gaming > 24:
        st.error("Jam bermain harus antara 0-24.")
        st.stop()

    if sleep < 0 or sleep > 24:
        st.error("Jam tidur harus antara 0-24.")
        st.stop()

    if device < 0 or device > 24:
        st.error("Jam penggunaan gadget harus antara 0-24.")
        st.stop()

    if sosial < 0 or sosial > 24:
        st.error("Jam sosialisasi harus antara 0-24.")
        st.stop()

    if attendance < 0 or attendance > 100:
        st.error("Kehadiran harus antara 0-100%.")
        st.stop()

    if study + gaming + sleep + sosial > 24:
        st.error("Total jam belajar, bermain, tidur, dan sosialisasi tidak boleh lebih dari 24 jam.")
        st.stop()

    if device < gaming:
        st.error("Jam penggunaan gadget tidak boleh lebih kecil dari jam bermain game.")
        st.stop()

    if study > 7 and sleep <= 6:
        stress = "High"
    elif gaming > 5:
        stress = "Low"
    else:
        stress = "Medium"

    addiction = -0.0024 + (1.4820 * gaming) + (0.5101 * device)

    st.info(f"Skor Kecanduan : {addiction:.2f}")
    st.info(f"Tingkat Stress : {stress}")

    input_df = pd.DataFrame(
        [[study, gaming, sleep, attendance, device, sosial, genre, addiction, stress]],
        columns=[
            "study_hours",
            "gaming_hours",
            "sleep_hours",
            "attendance",
            "device_usage",
            "social_activity",
            "gaming_genre",
            "addiction_score",
            "stress_level",
        ],
    )

    input_df["stress_level"] = le_stress.transform(input_df["stress_level"])
    input_df["gaming_genre"] = le_genre.transform(input_df["gaming_genre"])

    input_scaled = scaler.transform(input_df)

    hasil_encoded = model.predict(input_scaled)
    hasil = le_target.inverse_transform(hasil_encoded)[0]

    if hasil == "Lulus":
        st.success("Hasil Prediksi : Mahasiswa Berpotensi LULUS")
    else:
        st.error("Hasil Prediksi : Mahasiswa Berpotensi TIDAK LULUS")
