import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

class EmotionFusion:
    """
    A class to combine facial emotion detection with heart rate emotion prediction
    using multiple fusion methods.
    """
    
    def __init__(self, method='weighted', face_weight=0.7, heart_weight=0.3):
        """
        Initialize the fusion model.
        
        Parameters:
        -----------
        method : str
            Fusion method to use ('weighted', 'ensemble', 'rule_based')
        face_weight : float
            Weight given to facial emotion (for weighted method)
        heart_weight : float
            Weight given to heart rate-based emotion (for weighted method)
        """
        self.method = method
        self.face_weight = face_weight
        self.heart_weight = heart_weight
        self.ensemble_model = None
        self.emotion_mapping = {}
        self.emotion_history = []
        
    def train_ensemble(self, log_file, ground_truth_file=None):
        """
        Train the ensemble model using historical data.
        
        Parameters:
        -----------
        log_file : str
            Path to the log file with facial and heart emotions
        ground_truth_file : str, optional
            Path to file with ground truth emotions (if available)
        """
        try:
            # Load the emotion log data
            log_df = pd.read_excel(log_file)
            
            # If we have ground truth data, use it
            if ground_truth_file:
                truth_df = pd.read_csv(ground_truth_file)
                y_true = truth_df['TrueEmotion'].values
            else:
                # Otherwise, use facial emotion as approximate ground truth
                # (We'll improve this as we collect more data)
                y_true = log_df['Facial Emotion'].values
            
            # Create one-hot encoder for the emotions
            emotions = list(set(log_df['Facial Emotion'].tolist() + 
                               log_df['Heart Emotion'].tolist()))
            self.emotions = emotions
            
            # Maps emotion names to indices
            self.emotion_mapping = {emotion: i for i, emotion in enumerate(emotions)}
            
            # Create features: face emotion (one-hot), heart rate, heart emotion (one-hot)
            face_features = np.zeros((len(log_df), len(emotions)))
            heart_features = np.zeros((len(log_df), len(emotions)))
            
            for i, (_, row) in enumerate(log_df.iterrows()):
                if row['Facial Emotion'] in self.emotion_mapping:
                    face_features[i, self.emotion_mapping[row['Facial Emotion']]] = 1
                if row['Heart Emotion'] in self.emotion_mapping:
                    heart_features[i, self.emotion_mapping[row['Heart Emotion']]] = 1
            
            # Combine all features
            X_combined = np.column_stack([
                face_features,
                log_df['Heart Rate'].values.reshape(-1, 1),
                heart_features
            ])
            
            # Train the ensemble model
            self.ensemble_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ensemble_model.fit(X_combined, y_true)
            
            print("Ensemble model trained successfully!")
            return True
            
        except Exception as e:
            print(f"Error training ensemble model: {e}")
            return False
    
    def predict_emotion(self, face_emotion, heart_rate, heart_emotion, 
                       face_confidence=None, heart_confidence=None):
        """
        Predict the combined emotion using the selected fusion method.
        
        Parameters:
        -----------
        face_emotion : str
            Detected facial emotion
        heart_rate : float
            Current heart rate
        heart_emotion : str
            Predicted emotion from heart rate
        face_confidence : float, optional
            Confidence score for facial emotion (0-1)
        heart_confidence : float, optional
            Confidence score for heart emotion (0-1)
            
        Returns:
        --------
        str
            Predicted combined emotion
        """
        # Add to history for temporal analysis
        self.emotion_history.append({
            'face_emotion': face_emotion,
            'heart_rate': heart_rate,
            'heart_emotion': heart_emotion,
            'face_confidence': face_confidence,
            'heart_confidence': heart_confidence
        })
        
        # Keep only the last 10 records
        if len(self.emotion_history) > 10:
            self.emotion_history.pop(0)
        
        # Use selected fusion method
        if self.method == 'weighted':
            return self._weighted_fusion(face_emotion, heart_emotion, 
                                        face_confidence, heart_confidence)
        
        elif self.method == 'ensemble' and self.ensemble_model is not None:
            return self._ensemble_fusion(face_emotion, heart_rate, heart_emotion)
        
        elif self.method == 'rule_based':
            return self._rule_based_fusion(face_emotion, heart_rate, heart_emotion)
        
        elif self.method == 'temporal':
            return self._temporal_fusion()
        
        # Default fallback
        return self._weighted_fusion(face_emotion, heart_emotion, 
                                    face_confidence, heart_confidence)
    
    def _weighted_fusion(self, face_emotion, heart_emotion, 
                         face_confidence=None, heart_confidence=None):
        """Simple weighted decision between face and heart emotions."""
        # If confidences are provided, use them as weights
        if face_confidence is not None and heart_confidence is not None:
            face_weight = face_confidence
            heart_weight = heart_confidence
        else:
            face_weight = self.face_weight
            heart_weight = self.heart_weight
            
        # If both emotions agree, return that emotion
        if face_emotion == heart_emotion:
            return face_emotion
            
        # If emotions differ, use weighted decision
        if face_weight > heart_weight:
            return face_emotion
        else:
            return heart_emotion
    
    def _ensemble_fusion(self, face_emotion, heart_rate, heart_emotion):
        """Use the trained ensemble model to predict the combined emotion."""
        # Create feature vector for prediction
        face_features = np.zeros(len(self.emotions))
        heart_features = np.zeros(len(self.emotions))
        
        if face_emotion in self.emotion_mapping:
            face_features[self.emotion_mapping[face_emotion]] = 1
        if heart_emotion in self.emotion_mapping:
            heart_features[self.emotion_mapping[heart_emotion]] = 1
            
        X = np.concatenate([
            face_features,
            np.array([heart_rate]),
            heart_features
        ]).reshape(1, -1)
        
        # Make prediction
        return self.ensemble_model.predict(X)[0]
    
    def _rule_based_fusion(self, face_emotion, heart_rate, heart_emotion):
        """
        Rule-based fusion using domain knowledge about emotions.
        
        This method implements rules like:
        - High heart rate + neutral face = potential hidden stress
        - Low heart rate + happy face = relaxed happiness
        etc.
        """
        # Rule 1: High heart rate overrides neutral face
        if face_emotion == "Neutral" and heart_rate > 90:
            return heart_emotion
            
        # Rule 2: Very high heart rate + any negative emotion intensifies it
        if heart_rate > 100 and face_emotion in ["Angry", "Fearful", "Sad"]:
            return face_emotion  # Prioritize the facial emotion
            
        # Rule 3: Low heart rate makes "Happy" more likely to be "Neutral"
        if face_emotion == "Happy" and heart_rate < 65 and heart_emotion == "Neutral":
            return "Neutral"
            
        # Rule 4: Surprise with high heart rate is more likely fear
        if face_emotion == "Surprised" and heart_rate > 90 and heart_emotion == "Fearful":
            return "Fearful"
            
        # Default to facial emotion for most cases
        return face_emotion
    
    def _temporal_fusion(self):
        """
        Analyze emotion patterns over time to reduce noise and detect emotional transitions.
        Uses the last 10 emotion readings from history.
        """
        if len(self.emotion_history) < 3:
            # Not enough history, use most recent facial emotion
            return self.emotion_history[-1]['face_emotion']
            
        # Count frequency of recent facial emotions
        face_emotions = [entry['face_emotion'] for entry in self.emotion_history]
        face_counter = Counter(face_emotions)
        most_common_face = face_counter.most_common(1)[0][0]
        
        # Check if heart emotions have been consistent
        heart_emotions = [entry['heart_emotion'] for entry in self.emotion_history]
        heart_counter = Counter(heart_emotions)
        most_common_heart = heart_counter.most_common(1)[0][0]
        
        # Calculate average heart rate
        avg_heart_rate = sum(entry['heart_rate'] for entry in self.emotion_history) / len(self.emotion_history)
        
        # If facial emotion has been stable, trust it more
        if face_counter[most_common_face] >= len(self.emotion_history) * 0.6:
            return most_common_face
            
        # If heart emotion has been stable with changing facial expressions,
        # the person might be masking emotions
        if heart_counter[most_common_heart] >= len(self.emotion_history) * 0.7 and len(face_counter) > 2:
            return most_common_heart
            
        # Default to most recent weighted decision
        latest = self.emotion_history[-1]
        return self._weighted_fusion(
            latest['face_emotion'], 
            latest['heart_emotion'],
            latest.get('face_confidence'),
            latest.get('heart_confidence')
        )
    
    def save_model(self, filename="emotion_fusion_model.pkl"):
        """Save the fusion model to a file."""
        if self.ensemble_model:
            joblib.dump({
                'ensemble_model': self.ensemble_model,
                'emotion_mapping': self.emotion_mapping,
                'method': self.method,
                'face_weight': self.face_weight,
                'heart_weight': self.heart_weight,
                'emotions': self.emotions
            }, filename)
            print(f"Model saved to {filename}")
            return True
        return False
    
    def load_model(self, filename="emotion_fusion_model.pkl"):
        """Load a fusion model from a file."""
        try:
            data = joblib.load(filename)
            self.ensemble_model = data['ensemble_model']
            self.emotion_mapping = data['emotion_mapping']
            self.method = data.get('method', self.method)
            self.face_weight = data.get('face_weight', self.face_weight)
            self.heart_weight = data.get('heart_weight', self.heart_weight)
            self.emotions = data.get('emotions', list(self.emotion_mapping.keys()))
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# Example usage:
if __name__ == "__main__":
    # Create a fusion model with default weighted method
    fusion = EmotionFusion(method='weighted', face_weight=0.7, heart_weight=0.3)
    
    # Example prediction
    combined_emotion = fusion.predict_emotion(
        face_emotion="Happy", 
        heart_rate=85, 
        heart_emotion="Excited",
        face_confidence=0.8,
        heart_confidence=0.6
    )
    
    print(f"Combined emotion: {combined_emotion}")
    
    # If you have log data, you can train the ensemble model
    # fusion.train_ensemble("emotion_log.xlsx")
    
    # Then switch to using the ensemble method
    # fusion.method = 'ensemble'