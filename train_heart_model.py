import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# Load CSV
df = pd.read_csv('heart_rate_emotion_dataset.csv')

# Drop missing or irrelevant data
df = df.dropna()

# Features and labels
X = df[['HeartRate']]  # Capital H
y = df['Emotion']      # Capital E

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for future decoding
joblib.dump(label_encoder, 'label_encoder.pkl')

# Normalize heart rate values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'heart_scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model
joblib.dump(model, 'heart_model.pkl')
print("✅ Model, scaler, and label encoder saved!")
