import cv2
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from emotion_fusion import EmotionFusion  # Import our new fusion module

# Load models and tools with error handling
try:
    # Use cv2.data for cascade classifier path
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the complete model that includes both architecture and weights
    emotion_model = load_model('model.h5')
    
    # Load heart analysis models
    heart_model = joblib.load('heart_model.pkl')
    scaler = joblib.load('heart_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Initialize the emotion fusion module
    # You can change the method to 'ensemble', 'rule_based', or 'temporal'
    fusion = EmotionFusion(method='rule_based')
    
    # Try to load a pre-trained fusion model if it exists
    try:
        fusion.load_model()
        print("✅ Fusion model loaded successfully!")
    except:
        print("ℹ️ No fusion model found. Will use default fusion method.")
    
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# Emotion labels - match exactly with the original training code
emotion_labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load heart rate values with fallback
try:
    # Read the heart rate data from CSV
    heart_df = pd.read_csv('live_heart_data.csv')
    
    # Check if there are column names
    if 'HeartRate' in heart_df.columns:
        # If the CSV has a 'HeartRate' column, use that
        heart_values = heart_df['HeartRate'].tolist()
    else:
        # Otherwise, use the first column (assuming it contains heart rates)
        heart_values = heart_df.iloc[:, 0].tolist()
    
    print(f"✅ Loaded {len(heart_values)} heart rate values from CSV file")
    
    # Initialize a counter for cycling through available heart rate entries
    heart_index = 0
    
    # Calculate the time interval between readings to simulate real-time acquisition
    # (This will determine how often we move to the next heart rate entry)
    if len(heart_values) > 1:
        # For example, if we have 60 readings and want them to last for 1 minute,
        # we would update every second
        update_interval = 30  # Update heart rate every 30 frames (about once per second at 30 FPS)
    else:
        # If only one value, we'll just keep using it
        update_interval = 1000000  # A very large number so it never updates
        
    # Frame counter for heart rate updates
    frame_counter = 0
    
    # Current heart rate value
    current_heart_rate = heart_values[0]
    
except Exception as e:
    print(f"⚠️ Could not load heart rate data: {e}")
    # Default values as a fallback
    heart_values = [75, 80, 65, 90, 72, 78, 82, 68]
    heart_index = 0
    update_interval = 30
    frame_counter = 0
    current_heart_rate = heart_values[0]
    print("Using default heart rate values instead")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit(1)

# Output log
log_data = []

print("🚀 Press 'q' to quit and save log.")
print("🔄 Press 't' to toggle through fusion methods (weighted, ensemble, rule_based, temporal).")
print("🧠 Press 'l' to train fusion model from current log data.")

# List of available fusion methods for toggling
fusion_methods = ['weighted', 'ensemble', 'rule_based', 'temporal']
current_method_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    # Update frame counter for heart rate timing
    frame_counter += 1
    
    # Check if it's time to update the heart rate value
    if frame_counter >= update_interval:
        # Reset frame counter
        frame_counter = 0
        
        # Update to the next heart rate in the list
        heart_index = (heart_index + 1) % len(heart_values)
        current_heart_rate = heart_values[heart_index]
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        facial_emotion = "No Face"
        emotion_confidence = 0.0

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum(roi_gray) != 0:
                # Preprocess the face image the same way as in training
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                # Make prediction with verbose=0 to suppress output
                prediction = emotion_model.predict(roi, verbose=0)[0]
                maxindex = int(np.argmax(prediction))
                label = emotion_labels[maxindex]
                facial_emotion = label
                # Get confidence score for the prediction
                emotion_confidence = prediction[maxindex]

                # Draw box and label on webcam
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_with_conf = f"{label} ({emotion_confidence:.2f})"
                cv2.putText(frame, label_with_conf, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            break  # Only analyze first face
        
        # Use the current heart rate value
        heart_rate = current_heart_rate

        # Predict emotion from heart rate
        try:
            X = scaler.transform([[heart_rate]])
            predicted_emotion_index = heart_model.predict(X)[0]
            predicted_emotion = label_encoder.inverse_transform([predicted_emotion_index])[0]
            
            # Get confidence score for heart prediction (if possible)
            # Note: MLPClassifier doesn't provide confidence by default
            # so we'll use a placeholder value
            heart_confidence = 0.7  # Placeholder
        except Exception as e:
            print(f"❌ Heart prediction error: {e}")
            predicted_emotion = "Unknown"
            heart_confidence = 0.0

        # Use fusion model to combine predictions
        combined_emotion = fusion.predict_emotion(
            face_emotion=facial_emotion,
            heart_rate=heart_rate,
            heart_emotion=predicted_emotion,
            face_confidence=emotion_confidence,
            heart_confidence=heart_confidence
        )

        # Logging
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_data.append([timestamp, facial_emotion, heart_rate, predicted_emotion, combined_emotion])

        # Display info
        cv2.putText(frame, f"Face: {facial_emotion} ({emotion_confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Heart: {heart_rate} BPM -> {predicted_emotion}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"COMBINED: {combined_emotion} (using {fusion.method})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add key controls info
        cv2.putText(frame, "'q': quit, 't': toggle fusion, 'l': learn from log", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
    
    cv2.imshow("Emotion Monitor", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        # Toggle through fusion methods
        current_method_index = (current_method_index + 1) % len(fusion_methods)
        fusion.method = fusion_methods[current_method_index]
        print(f"Switched to fusion method: {fusion.method}")
    elif key == ord('l'):
        # If we have enough log data, try to train the ensemble model
        if len(log_data) > 10:
            # Save current log to a temporary file
            temp_df = pd.DataFrame(log_data, 
                                 columns=['Timestamp', 'Facial Emotion', 'Heart Rate', 
                                         'Heart Emotion', 'Combined Emotion'])
            temp_df.to_excel("temp_log.xlsx", index=False)
            
            # Train the ensemble model using this data
            if fusion.train_ensemble("temp_log.xlsx"):
                fusion.method = 'ensemble'  # Switch to using the trained model
                print("✅ Trained and switched to ensemble fusion model")
            else:
                print("❌ Could not train ensemble model")
        else:
            print("⚠️ Not enough log data to train model (need at least 10 entries)")

# Save log to Excel with error handling
try:
    df = pd.DataFrame(log_data, 
                     columns=['Timestamp', 'Facial Emotion', 'Heart Rate', 
                             'Heart Emotion', 'Combined Emotion'])
    df.to_excel("emotion_log.xlsx", index=False)
    print("✅ Log saved to emotion_log.xlsx")
    
    # Save the fusion model
    fusion.save_model()
except Exception as e:
    print(f"❌ Error saving Excel file: {e}")
    # Backup save as CSV
    try:
        df.to_csv("emotion_log_backup.csv", index=False)
        print("✅ Backup log saved to emotion_log_backup.csv")
    except:
        print("❌ Failed to save log")

# Release resources
cap.release()
cv2.destroyAllWindows()