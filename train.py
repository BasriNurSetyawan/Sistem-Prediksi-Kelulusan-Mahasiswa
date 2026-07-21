import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load dataset (pastikan file CSV tersedia)
df = pd.read_csv('Gaming_Academic_Performance_updated.csv').dropna()

# Jika di dataset asli belum ada kolom 'assignment_completion', 
# kita buatkan simulasi kolomnya berdasarkan nilai grades (opsional, sesuaikan jika kolom sudah ada di CSV)
if 'assignment_completion' not in df.columns:
    # Contoh estimasi proporsi pengumpulan tugas berdasarkan nilai akhir
    df['assignment_completion'] = df['grades'].apply(lambda x: min(100.0, max(30.0, x + np.random.uniform(-10, 10))))

# Label target kelulusan (>= 60 Lulus, di bawah itu Tidak Lulus)
df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if x == 'Lulus' or (isinstance(x, (int, float)) and x >= 60) else 'Tidak Lulus')
# Perbaikan jika kolom grades sudah berupa kategori teks di dataset lu:
if 'grades' in df.columns and df['grades'].dtype == 'object':
    df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if str(x).lower() in ['lulus', 'pass', 'b', 'a'] else 'Tidak Lulus')

# 2. Tentukan Fitur Input & Target
features = [
    'study_hours', 'gaming_hours', 'sleep_hours', 
    'attendance', 'assignment_completion', 'device_usage', 
    'social_activity', 'addiction_score', 'stress_level'
]

X = df[features].copy()
y = df['status_lulus']

# 3. Encoding Variabel Kategori (Stress Level)
le_stress = LabelEncoder()
le_target = LabelEncoder()

X['stress_level'] = le_stress.fit_transform(X['stress_level'])
y_encoded = le_target.fit_transform(y)

# 4. Normalisasi Angka dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data & Training Model Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Simpan Komponen ke File .pkl (Tanpa le_genre.pkl karena sudah dihapus)
joblib.dump(model, 'model_akademik.pkl')
joblib.dump(scaler, 'scaler_akademik.pkl')
joblib.dump(le_stress, 'le_stress.pkl')
joblib.dump(le_target, 'le_target.pkl')

print("Training selesai! File .pkl baru berhasil diperbarui.")