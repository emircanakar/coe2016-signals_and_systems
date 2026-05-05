import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. Load the extracted features dataset
df = pd.read_csv("extracted_features.csv")


# 2. Extract the emotion (label) from the filename
# We will mark corrupted or unrecognized files as "Unknown"
def get_emotion(filename):
    filename = filename.lower()
    if "happy" in filename:
        return "Happy"
    elif "angry" in filename or "furious" in filename:
        return "Angry"
    elif "sad" in filename:
        return "Sad"
    elif "surprised" in filename:
        return "Surprised"
    elif "neutral" in filename:
        return "Neutral"
    else:
        return "Unknown"


df['Emotion'] = df['Filename'].apply(get_emotion)

# Clean the dataset by removing rows with "Unknown" emotions
df = df[df['Emotion'] != 'Unknown']

# 3. Separate features (X) and the target labels (y)
feature_cols = ["ZCR", "RMS", "Pitch"] + [f"MFCC_{i}" for i in range(1, 14)]
X = df[feature_cols]
y = df['Emotion']

# 4. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and train the Random Forest Classifier
print("Training the model, please wait...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the model using the unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("-" * 40)
print(f"PHASE 1 - MODEL ACCURACY: {accuracy * 100:.2f}%")
print("-" * 40)

# Detailed analysis to see which emotions are confused by the model
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))
