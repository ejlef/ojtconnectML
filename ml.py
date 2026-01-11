import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import os

# ===== CONFIG =====
SERVICE_ACCOUNT = "ojtconnect-cca5b-firebase-adminsdk-fbsvc-39a3f4ccb7.json"
ML_COLLECTION = "ml_predictions"
SCHEDULED_TIME_STR = "08:00:00"  # Scheduled check-in time
MIN_RECORDS_FOR_ML = 10  # minimum rows needed to train real model

# ===== FIRESTORE INIT =====
if not os.path.exists(SERVICE_ACCOUNT):
    raise FileNotFoundError(f"{SERVICE_ACCOUNT} not found. Place it in project folder.")

cred = credentials.Certificate(SERVICE_ACCOUNT)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ===== FETCH ATTENDANCE =====
attendance_docs = list(db.collection('attendance').stream())
attendance_data = []
for doc in attendance_docs:
    d = doc.to_dict()
    date_val = d.get('date')
    time_val = d.get('time')

    # Safe parsing
    if hasattr(date_val, 'strftime'):
        date_val = date_val.strftime("%Y-%m-%d")
    if hasattr(time_val, 'strftime'):
        time_val = time_val.strftime("%H:%M:%S")

    attendance_data.append({
        'studentId': d.get('studentId', ''),
        'studentName': d.get('studentName', ''),
        'date': date_val,
        'time': time_val,
        'status': d.get('status', ''),
    })

df_att = pd.DataFrame(attendance_data)
if df_att.empty:
    print("⚠️ No attendance data found. Creating dummy data.")
    students = [f"S{str(i).zfill(3)}" for i in range(1, 16)]
    dates = pd.date_range("2026-01-01", periods=15)
    dummy_data = []
    for student in students:
        for date in dates:
            check_in_hour = np.random.choice([8,8,8,9,10])
            status = "check-in" if np.random.rand() > 0.1 else "absent"
            dummy_data.append({
                "studentId": student,
                "studentName": f"Student {student}",
                "date": date.strftime("%Y-%m-%d"),
                "time": f"{check_in_hour}:{np.random.randint(0,60):02d}:00" if status=="check-in" else "",
                "status": status
            })
    df_att = pd.DataFrame(dummy_data)

# ===== FEATURE ENGINEERING =====
df_att['date'] = pd.to_datetime(df_att['date'], errors='coerce')
scheduled_time = datetime.strptime(SCHEDULED_TIME_STR, "%H:%M:%S").time()
df_att['checkInTime'] = pd.to_datetime(df_att['time'], errors='coerce').dt.time

# Minutes late
df_att['minutesLate'] = df_att.apply(
    lambda x: max(0, (datetime.combine(datetime.today(), x['checkInTime']) -
                       datetime.combine(datetime.today(), scheduled_time)).seconds // 60)
    if pd.notnull(x['checkInTime']) else 0,
    axis=1
)
df_att['lateFlag'] = (df_att['minutesLate'] > 0).astype(int)
df_att['absentFlag'] = (df_att['status'] != "check-in").astype(int)

# Aggregate per student
student_features = df_att.groupby('studentId').agg({
    'minutesLate': 'mean',
    'lateFlag': 'sum',
    'absentFlag': 'sum'
}).reset_index()

# ===== FETCH OJT HOURS & Reports =====
teams_docs = list(db.collection('teams').stream())
hours_data = []

for doc in teams_docs:
    d = doc.to_dict()
    students_list = d.get('students', [])
    if not students_list:
        continue  # skip teams without students
    for student in students_list:
        sid = student.get('id') or student.get('studentId')
        if sid:
            hours_data.append({
                'studentId': sid,
                'ojtHoursCompleted': student.get('completedHours', 0),
                'requiredHours': d.get('requiredHours', 0),
                'feedbackScore': student.get('feedbackScore', 0),
                'reportSubmissionRate': student.get('reportSubmissionRate', 0)
            })

df_hours = pd.DataFrame(hours_data)

# Safe fallback if empty
if df_hours.empty:
    df_hours = pd.DataFrame({
        'studentId': student_features['studentId'],
        'ojtHoursCompleted': 0,
        'requiredHours': 0,
        'feedbackScore': 0,
        'reportSubmissionRate': 0
    })

# Merge
student_features = student_features.merge(df_hours, on='studentId', how='left')

# ===== ML PREP =====
X = student_features[['ojtHoursCompleted', 'requiredHours', 'lateFlag', 'absentFlag',
                      'reportSubmissionRate', 'feedbackScore']].fillna(0).values

# Targets: classify risk level
def label_risk(row):
    if row['absentFlag'] > 3 or row['lateFlag'] > 5:
        return "High"
    elif row['absentFlag'] > 1 or row['lateFlag'] > 2:
        return "Medium"
    else:
        return "Low"

student_features['riskLabel'] = student_features.apply(label_risk, axis=1)

y = student_features['riskLabel']

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===== TRAIN OR LOAD MODEL =====
MODEL_FILE = "risk_model.pkl"

def train_or_load_model(X, y, filename):
    if os.path.exists(filename):
        model = joblib.load(filename)
        return model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    joblib.dump(model, filename)
    return model

model = train_or_load_model(X, y_encoded, MODEL_FILE)

# ===== PREDICTIONS =====
pred_probs = model.predict_proba(X)
pred_labels = le.inverse_transform(model.predict(X))

student_features['riskPrediction'] = pred_labels
student_features['riskProbability'] = pred_probs.max(axis=1)

# ===== UPLOAD TO FIRESTORE =====
print("⬆️ Uploading ML predictions to Firestore...")
id_to_name = df_att.groupby('studentId')['studentName'].first().to_dict()

for _, row in student_features.iterrows():
    doc_id = row['studentId']
    student_name = id_to_name.get(doc_id, '')
    db.collection(ML_COLLECTION).document(doc_id).set({
        'studentId': doc_id,
        'studentName': student_name,
        'riskLabel': row['riskLabel'],
        'riskPrediction': row['riskPrediction'],
        'riskProbability': float(row['riskProbability']),
        'updatedAt': firestore.SERVER_TIMESTAMP
    })

print("✅ ML predictions uploaded successfully!")
