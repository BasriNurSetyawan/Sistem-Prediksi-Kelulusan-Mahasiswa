import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Mulai proses training model...")

df = pd.read_csv('Gaming_Academic_Performance_updated.csv').dropna()

if 'assignment_completion' not in df.columns:
    df['assignment_completion'] = df['grades'].apply(lambda x: min(100.0, max(30.0, float(x) + 5.0)) if isinstance(x, (int, float, str)) and str(x).replace('.','',1).isdigit() else 75.0)

if 'grades' in df.columns and df['grades'].dtype == 'object':
    df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if str(x).lower() in ['lulus', 'pass', 'b', 'a'] else 'Tidak Lulus')
else:
    df['status_lulus'] = df['grades'].apply(lambda x: 'Lulus' if float(x) >= 60 else 'Tidak Lulus')

features = [
    'study_hours', 'gaming_hours', 'sleep_hours', 
    'attendance', 'assignment_completion', 'device_usage', 
    'social_activity', 'addiction_score', 'stress_level'
]

X = df[features].copy()
y = df['status_lulus']

le_stress = LabelEncoder()
le_target = LabelEncoder()
X['stress_level'] = le_stress.fit_transform(X['stress_level'])
y_encoded = le_target.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_encoded, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, output_dict=True)

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

metrics = {
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1_Score': f1,
    'Classification_Report': cr,
    'Confusion_Matrix': cm.tolist()
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

feature_importance.to_csv('feature_importance.csv', index=False)

joblib.dump(model, 'model_akademik.pkl')
joblib.dump(scaler, 'scaler_akademik.pkl')
joblib.dump(le_stress, 'le_stress.pkl')
joblib.dump(le_target, 'le_target.pkl')

print("\n=== HASIL EVALUASI ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance.to_string(index=False))
print("\n=== FILE TERSIMPAN ===")
print("✅ model_akademik.pkl\n✅ scaler_akademik.pkl\n✅ le_target.pkl\n✅ le_stress.pkl\n✅ metrics.json\n✅ feature_importance.csv")
print("Proses Selesai!")