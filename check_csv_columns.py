import pandas as pd

# Load your heart rate dataset
df = pd.read_csv("heart_rate_emotion_dataset.csv")

# Print column names and sample data
print("🔥 Columns found in CSV:")
print(df.columns)

print("\n📊 Sample data:")
print(df.head())
