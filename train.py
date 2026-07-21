import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('Gaming_Academic_Performance_updated.csv')
df = df.dropna()

# Buat target kelulusan
df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if x >= 60 else 'Tidak Lulus')

# Fitur yang digunakan sekarang lebih banyak
X = df[['study_hours', 'gaming_hours', 'sleep_hours', 'attendance', 'device_usage', 'social_activity', 'gaming_genre', 'addiction_score', 'stress_level']].copy()
y = df['status_lulus']

# Ubah teks jadi angka buat AI
le_stress = LabelEncoder()
X['stress_level'] = le_stress.fit_transform(X['stress_level'])

le_genre = LabelEncoder()
X['gaming_genre'] = le_genre.fit_transform(X['gaming_genre'])

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalisasi data angka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Latih AI
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Simpan semua model dan encoder
joblib.dump(model, 'model_akademik.pkl')
joblib.dump(scaler, 'scaler_akademik.pkl')
joblib.dump(le_stress, 'le_stress.pkl')
joblib.dump(le_genre, 'le_genre.pkl')
joblib.dump(le_target, 'le_target.pkl')

print("Model baru berhasi dibuat dan disimpan!")