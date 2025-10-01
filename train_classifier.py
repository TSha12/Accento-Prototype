import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset/lang/wav_names.csv")

features = []
labels = []

# Extract MFCC features from each audio file
for i, row in df.iterrows():
    file_path = os.path.join("dataset/lang", row["File Name"])
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feat = mfcc.mean(axis=1)
    features.append(feat)
    labels.append(row["Accent"])

# Convert features and labels into numpy arrays
X = np.array(features)
y = np.array(labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set and print performance
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/accent_classifier.pkl")
print("Model saved to models/accent_classifier.pkl")

# Save features and labels for visualization
np.save("data/features.npy", X)
np.save("data/labels.npy", y)

# Save the test data for further analysis
os.makedirs("data", exist_ok=True)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
print("Test data saved to data/ folder")
