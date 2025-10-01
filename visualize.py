import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

# Create plots directory
os.makedirs("plots", exist_ok=True)

# Load dataset for accent distribution and pairplot
df = pd.read_csv("dataset/lang/wav_names.csv")

# Plot 1 – Accent Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Accent')
plt.title("Accent Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/accent_distribution.png")
plt.show(block=False)

input("Press Enter to view MFCC Pairplot...")

# Load extracted features and labels
features = np.load("data/features.npy")
labels = np.load("data/labels.npy")

# Plot 2 – MFCC Pairplot (limit to 100 samples)
sample_size = min(100, len(labels))
sample_indices = np.random.choice(len(labels), sample_size, replace=False)

feature_sample = features[sample_indices]
label_sample = labels[sample_indices]

feature_df = pd.DataFrame(feature_sample, columns=[f'MFCC_{i+1}' for i in range(13)])
feature_df['Accent'] = label_sample

sns.pairplot(feature_df, hue="Accent", diag_kind="kde", palette="bright")
plt.suptitle("MFCC Feature Pairplot", y=1.02)
plt.tight_layout()
plt.savefig("plots/mfcc_pairplot.png")
plt.show(block=False)

input("Press Enter to view PCA and t-SNE plots...")

# Load trained model and test data
clf = joblib.load("models/accent_classifier.pkl")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# PCA Plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8, 6))
for accent in np.unique(y_test):
    idx = y_test == accent
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=accent)
plt.title("PCA Projection of Test Data")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/pca_projection.png")
plt.show(block=False)

input("Press Enter to view t-SNE plot...")

# t-SNE Plot
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_test)

plt.figure(figsize=(8, 6))
for accent in np.unique(y_test):
    idx = y_test == accent
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=accent)
plt.title("t-SNE Projection of Test Data")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/tsne_projection.png")
plt.show(block=False)

input("Press Enter to view Feature Importance...")

# Feature Importance Plot
if hasattr(clf, 'feature_importances_'):
    importances = clf.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f'MFCC_{i+1}' for i in range(len(importances))], y=importances)
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("MFCC Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.show()
else:
    print("The classifier does not provide feature importances.")
