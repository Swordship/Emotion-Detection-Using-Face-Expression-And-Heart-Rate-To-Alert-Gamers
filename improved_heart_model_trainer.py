import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import joblib
import os
import time


# Enable GPU acceleration and memory growth to avoid taking all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s). Enabling memory growth.")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Using GPU for training")
else:
    print("No GPU found. Using CPU for training")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path, normalize=True, sample_fraction=1.0):
    """
    Load and preprocess the heart rate emotion dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing heart rate and emotion data
    normalize : bool
        Whether to normalize the heart rate values
    sample_fraction : float
        Fraction of data to use (can reduce this for faster training)
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    # Load CSV
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # If sampling, take only a fraction of the data
    if sample_fraction < 1.0:
        original_size = len(df)
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"Sampled {len(df)} rows from {original_size} ({sample_fraction*100:.1f}%) for faster training")
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Drop missing values
    original_size = len(df)
    df = df.dropna()
    if len(df) < original_size:
        print(f"Dropped {original_size - len(df)} rows with missing values")
    
    # Features and labels
    X = df[['HeartRate']]
    y = df['Emotion']
    
    # Show class distribution
    class_distribution = y.value_counts()
    print("\nClass distribution:")
    for emotion, count in class_distribution.items():
        print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Normalize heart rate values if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Normalized heart rate range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    else:
        X_scaled = X.values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

# This should be moved to the top level of your file, outside any functions
class TFModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)
        
    def predict_proba(self, X):
        return self.model.predict(X)

# Then your function should be modified to use this global class:
def train_mlp_model_gpu(X_train, y_train, X_test, y_test, label_encoder):
    """Train a MLP model using TensorFlow with GPU acceleration."""
    print("\n=== Training MLP Model with GPU Acceleration ===")
    start_time = time.time()
    
    # Convert data to TensorFlow format
    n_classes = len(label_encoder.classes_)
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
    
    # Create an MLP model
    model = tf.keras.Sequential([
        # Input layer: Number of features (1 for heart rate)
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        # Hidden layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # Output layer: Number of emotions
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Learning rate reducer to improve training
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    
    # Train the model
    print("Training GPU-accelerated MLP model...")
    history = model.fit(
        X_train, y_train_one_hot,
        epochs=50,
        batch_size=256,  # Larger batch size for GPU
        validation_split=0.2,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating GPU MLP model...")
    _, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('MLP Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('MLP Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_gpu_training_history.png')
    plt.close()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"MLP model training completed in {training_time:.2f} seconds")
    
    # Use the global TFModelWrapper class instead of defining it locally
    mlp_model = TFModelWrapper(model)
    
    # Save the raw TensorFlow model
    os.makedirs('models', exist_ok=True)
    model.save('models/mlp_gpu_model')
    print("Saved TensorFlow MLP model to models/mlp_gpu_model")
    
    return mlp_model

def train_random_forest_model_reduced(X_train, y_train, X_test, y_test, label_encoder):
    """Train a Random Forest model with reduced parameter search for faster training."""
    print("\n=== Training Random Forest Model with Optimized Parameters ===")
    start_time = time.time()
    
    # Instead of grid search, use randomized search with fewer iterations
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create RandomizedSearchCV object - much faster than GridSearchCV
    print("Performing randomized search with reduced parameters...")
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),  # Use all CPU cores
        param_distributions=param_dist,
        n_iter=10,  # Only try 10 combinations instead of all possible
        cv=3,       # Use 3-fold instead of 5-fold CV for speed
        verbose=1,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    # Perform random search
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Plot feature importance (even though it's just one feature, for consistency)
    plt.figure(figsize=(10, 6))
    plt.bar(['Heart Rate'], best_rf.feature_importances_)
    plt.title('Feature Importance - Random Forest Model')
    plt.savefig('rf_feature_importance.png')
    plt.close()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Random Forest model training completed in {training_time:.2f} seconds")
    
    return best_rf

def prepare_sequence_data(X, y, sequence_length=5, step=1):
    """
    Prepare sequence data for LSTM model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature data (heart rate values)
    y : numpy.ndarray
        Target data (emotion labels)
    sequence_length : int
        Number of time steps in each sequence
    step : int
        Step size between sequences
    
    Returns:
    --------
    X_seq : numpy.ndarray
        Sequence data with shape (n_samples, sequence_length, n_features)
    y_seq : numpy.ndarray
        Target data corresponding to each sequence
    """
    X_seq, y_seq = [], []
    
    # X is already a 2D array with one feature (heart rate)
    # We need to create sequences of length sequence_length
    for i in range(0, len(X) - sequence_length, step):
        X_seq.append(X[i:i + sequence_length])
        # Use the label of the last time step in the sequence
        y_seq.append(y[i + sequence_length - 1])
    
    return np.array(X_seq), np.array(y_seq)

def train_bilstm_model_gpu(X_train, y_train, X_test, y_test, label_encoder, sequence_length=5):
    """Train a Bidirectional LSTM model with GPU acceleration."""
    print("\n=== Training Bidirectional LSTM Model with GPU Acceleration ===")
    start_time = time.time()
    
    # Try to use mixed precision for faster GPU training
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Using mixed precision for faster GPU training")
    except:
        print("Mixed precision not available, using default precision")
    
    # Prepare sequence data
    print(f"Preparing sequence data with length {sequence_length}...")
    X_train_seq, y_train_seq = prepare_sequence_data(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = prepare_sequence_data(X_test, y_test, sequence_length)
    
    print(f"Sequence training data shape: {X_train_seq.shape}")
    print(f"Sequence test data shape: {X_test_seq.shape}")
    
    # Get number of classes
    n_classes = len(label_encoder.classes_)
    
    # Create one-hot encoded targets for Keras
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_seq, num_classes=n_classes)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test_seq, num_classes=n_classes)
    
    # Build optimized Bi-LSTM model
    model = Sequential([
        # Input shape: (sequence_length, features)
        Bidirectional(LSTM(64, return_sequences=True), 
                     input_shape=(sequence_length, 1)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    # Compile model with higher batch size for GPU
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Learning rate reducer
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
    
    # Train model with increased batch size for GPU
    print("\nTraining GPU-accelerated Bi-LSTM model...")
    history = model.fit(
        X_train_seq, y_train_one_hot,
        epochs=10,
        batch_size=512,  # Much larger batch size for GPU acceleration
        validation_split=0.2,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating Bi-LSTM model...")
    _, accuracy = model.evaluate(X_test_seq, y_test_one_hot, verbose=0)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    
    # Get predictions
    y_pred_proba = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_seq, y_pred, target_names=label_encoder.classes_))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Bi-LSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Bi-LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bilstm_gpu_training_history.png')
    plt.close()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Bi-LSTM model training completed in {training_time:.2f} seconds")
    
    return model

def analyze_heart_rate_ranges(file_path):
    """Analyze heart rate ranges for different emotions."""
    print("\n=== Analyzing Heart Rate Ranges ===")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Basic statistics by emotion
    stats_by_emotion = df.groupby('Emotion')['HeartRate'].agg(['min', 'max', 'mean', 'std'])
    print("\nHeart Rate Statistics by Emotion:")
    print(stats_by_emotion)
    
    # Plot heart rate distributions by emotion
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Emotion', y='HeartRate', data=df)
    plt.title('Heart Rate Distributions by Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Heart Rate (BPM)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('heart_rate_by_emotion.png')
    
    # Plot heart rate density by emotion
    plt.figure(figsize=(14, 8))
    for emotion in df['Emotion'].unique():
        subset = df[df['Emotion'] == emotion]
        sns.kdeplot(subset['HeartRate'], label=emotion)
    
    plt.title('Heart Rate Density by Emotion')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('heart_rate_density.png')
    plt.close()
    
    # Create a lookup table for heart rate ranges and emotions
    emotion_ranges = {}
    for emotion in df['Emotion'].unique():
        subset = df[df['Emotion'] == emotion]
        # Get 5th and 95th percentiles for more robust ranges
        p05 = np.percentile(subset['HeartRate'], 5)
        p95 = np.percentile(subset['HeartRate'], 95)
        emotion_ranges[emotion] = {
            'min': p05,
            'max': p95,
            'mean': subset['HeartRate'].mean()
        }
    
    return emotion_ranges

def save_models(mlp_model, rf_model, bilstm_model, scaler, label_encoder, emotion_ranges=None):
    """
    Save all models and metadata for later use.
    
    Parameters:
    -----------
    mlp_model : TFModelWrapper or MLPClassifier
        Trained MLP model
    rf_model : RandomForestClassifier
        Trained Random Forest model
    bilstm_model : keras.Model
        Trained Bi-LSTM model
    scaler : StandardScaler
        Fitted scaler for heart rate normalization
    label_encoder : LabelEncoder
        Fitted label encoder for emotion classes
    emotion_ranges : dict, optional
        Dictionary with heart rate ranges by emotion
    """
    print("\n=== Saving Models and Metadata ===")
    
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save MLP model (this is already saved for TF model)
    joblib.dump(mlp_model, 'models/heart_mlp_model.pkl')
    print("Saved MLP model to models/heart_mlp_model.pkl")
    
    # Save Random Forest model
    joblib.dump(rf_model, 'models/heart_rf_model.pkl')
    print("Saved Random Forest model to models/heart_rf_model.pkl")
    
    # Save Bi-LSTM model (if it exists)
    if bilstm_model is not None:
        try:
            bilstm_model.save('models/heart_bilstm_model.h5')
            print("Saved Bi-LSTM model to models/heart_bilstm_model.h5")
        except Exception as e:
            print(f"Error saving Bi-LSTM model: {e}")
    
    # Save scaler
    joblib.dump(scaler, 'models/heart_scaler.pkl')
    print("Saved scaler to models/heart_scaler.pkl")
    
    # Save additional versions that will be used by the live emotion logger
    joblib.dump(scaler, 'heart_scaler.pkl')
    
    # Save label encoder
    joblib.dump(label_encoder, 'models/heart_label_encoder.pkl')
    print("Saved label encoder to models/heart_label_encoder.pkl")
    
    # Save additional versions that will be used by the live emotion logger
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save emotion ranges if provided
    if emotion_ranges:
        joblib.dump(emotion_ranges, 'models/emotion_heart_ranges.pkl')
        print("Saved emotion heart rate ranges to models/emotion_heart_ranges.pkl")
        
        # Also save as a readable JSON file
        import json
        with open('models/emotion_heart_ranges.json', 'w') as f:
            json.dump(emotion_ranges, f, indent=4)
        print("Saved emotion heart rate ranges to models/emotion_heart_ranges.json")
    
    # Save a simple model ready to use in the app (use Random Forest for best results)
    if rf_model is not None:
        heart_model = rf_model  # Random Forest typically gives best results for tabular data
    else:
        heart_model = mlp_model  # Fall back to MLP if RF is not available
    
    joblib.dump(heart_model, 'heart_model.pkl')
    print("Saved primary heart model to heart_model.pkl")
    
    # Save a bundled model for easier loading
    heart_model_bundle = {
        'model': heart_model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    joblib.dump(heart_model_bundle, 'heart_model_bundle.pkl')
    print("Saved bundled heart model to heart_model_bundle.pkl")

def compare_models(X_test, y_test, mlp_model, rf_model, bilstm_model=None, sequence_length=5, label_encoder=None):
    """Compare the performance of different models."""
    print("\n=== Model Comparison ===")
    
    # Predictions from MLP
    mlp_pred = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    
    # Predictions from Random Forest
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # Predictions from Bi-LSTM (if provided)
    bilstm_acc = None
    if bilstm_model is not None and label_encoder is not None:
        # Prepare sequence data
        X_test_seq, y_test_seq = prepare_sequence_data(X_test, y_test, sequence_length)
        n_classes = len(label_encoder.classes_)
        
        # Get predictions
        y_pred_proba = bilstm_model.predict(X_test_seq)
        bilstm_pred = np.argmax(y_pred_proba, axis=1)
        bilstm_acc = accuracy_score(y_test_seq, bilstm_pred)
    
    # Print results
    print("\nModel Accuracy Comparison:")
    print(f"MLP (GPU): {mlp_acc*100:.2f}%")
    print(f"Random Forest: {rf_acc*100:.2f}%")
    if bilstm_acc is not None:
        print(f"Bi-LSTM (GPU): {bilstm_acc*100:.2f}%")
    
    # Plot comparison
    models = ['MLP (GPU)', 'Random Forest']
    accuracies = [mlp_acc, rf_acc]
    
    if bilstm_acc is not None:
        models.append('Bi-LSTM (GPU)')
        accuracies.append(bilstm_acc)
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, [acc * 100 for acc in accuracies])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    for i, acc in enumerate(accuracies):
        plt.text(i, acc * 100 + 1, f"{acc*100:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Return the best model type
    best_acc = max(filter(None, [mlp_acc, rf_acc, bilstm_acc]))
    if best_acc == mlp_acc:
        return 'MLP (GPU)'
    elif best_acc == rf_acc:
        return 'Random Forest'
    else:
        return 'Bi-LSTM (GPU)'

def main():
    """Main function to run the heart rate emotion model training pipeline."""
    print("=== GPU-Accelerated Heart Rate Emotion Model Training ===")
    overall_start_time = time.time()
    
    # Check for data file
    file_path = 'heart_rate_emotion_dataset.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file '{file_path}' not found!")
        return
    
    # Ask if user wants to sample data for faster training
    use_sample = input("Do you want to use a smaller sample of data for faster training? (y/n): ").lower() == 'y'
    sample_fraction = 0.3 if use_sample else 1.0  # Use 30% of data or full dataset
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_preprocess_data(
        file_path, normalize=True, sample_fraction=sample_fraction
    )
    
    # Analyze heart rate ranges
    emotion_ranges = analyze_heart_rate_ranges(file_path)
    
    # Train MLP model with GPU
    mlp_model = train_mlp_model_gpu(X_train, y_train, X_test, y_test, label_encoder)
    
    # Train Random Forest model (CPU-based but optimized)
    rf_model = train_random_forest_model_reduced(X_train, y_train, X_test, y_test, label_encoder)
    
    # Ask if user wants to train Bi-LSTM model (takes longer but more accurate)
    train_lstm = input("\nDo you want to train a Bi-LSTM model? This takes longer but may be more accurate (y/n): ").lower() == 'y'
    bilstm_model = None
    if train_lstm:
        sequence_length = 5  # Default sequence length
        bilstm_model = train_bilstm_model_gpu(X_train, y_train, X_test, y_test, label_encoder, sequence_length)
    
    # Compare models
    best_model = compare_models(X_test, y_test, mlp_model, rf_model, bilstm_model, 
                              sequence_length=5 if train_lstm else None, 
                              label_encoder=label_encoder if train_lstm else None)
    print(f"\nBest performing model: {best_model}")
    
    # Save models
    save_models(mlp_model, rf_model, bilstm_model if train_lstm else None, 
              scaler, label_encoder, emotion_ranges)
    
    # Calculate total training time
    overall_time = time.time() - overall_start_time
    minutes = int(overall_time // 60)
    seconds = int(overall_time % 60)
    
    print("\n=== Training Complete! ===")
    print(f"Total training time: {minutes} minutes and {seconds} seconds")
    print("Models and data have been saved to the 'models' directory.")
    print("Key files are also saved in the main directory for easy use with the emotion logger.")
    print("\nYou can now run your emotion detection application with the trained models!")

if __name__ == "__main__":
    main()