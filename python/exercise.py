import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

folder_path = 'BicepsCurls'
target_length = 60
folds = 5
random_forest_filename = 'random_forest_model.pkl'
dtw_filename = 'dtw_reference.npy'


def resize_series_multidim(series, target_len):
    return np.array([
        np.interp(np.linspace(0, 1, target_len),
                  np.linspace(0, 1, len(series)),
                  series[:, i])
        for i in range(series.shape[1])
    ]).T


def load_and_process_csv(file_path, target_len):
    df = pd.read_csv(file_path)
    df = df[['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']]
    data = df.values
    data_resized = resize_series_multidim(data, target_len)
    stats = np.concatenate([data_resized.mean(axis=0), data_resized.std(axis=0)])
    flattened = data_resized.flatten()
    return flattened, stats, data_resized


X_stats, y, raw_sequences = [], [], []

for fname in os.listdir(folder_path):
    if not fname.endswith(".csv"):
        continue
    label = 0 if "incorecta" in fname.lower() else 1
    path = os.path.join(folder_path, fname)
    try:
        flattened, stats, raw = load_and_process_csv(path, target_length)
        X_stats.append(stats)
        raw_sequences.append(raw)
        y.append(label)
    except Exception as e:
        print(f"Error processing file {fname}: {e}")

X_stats = np.array(X_stats)
y = np.array(y)

print("\n=== Cross-Validation ===")
cv_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

cv_scores = cross_val_score(cv_model, X_stats, y, cv=folds)
print(f"Scores per fold ({folds}-fold): {cv_scores}")
print("Average CV Accuracy: {:.2f}".format(cv_scores.mean()))

plt.figure(figsize=(8, 5))
plt.plot(range(1, folds + 1), cv_scores, marker='o', linestyle='-', linewidth=2)
plt.ylim(0.7, 1.05)
plt.xticks(range(1, folds + 1))
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title(f"Accuracy per Fold ({folds}-fold)")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracies_per_fold.png")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_stats, y, test_size=0.2, random_state=42, stratify=y)

final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n=== Test Report ===")
print(classification_report(y_test, y_pred))
print(f"\nFinal test accuracy: {acc:.2f}")

train_accuracy = cv_scores.mean()
test_accuracy = acc

plt.figure(figsize=(6, 5))
plt.bar(['Train', 'Test'], [train_accuracy, test_accuracy], color=['skyblue', 'orange'], width=0.5)
plt.ylim(0.5, 1.05)
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy: Train vs Test")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("train_vs_test_bar.png")
plt.show()

joblib.dump(final_model, random_forest_filename)
print(f"Model saved to: {random_forest_filename}")

for seq, label in zip(raw_sequences, y):
    if label == 1:
        np.save(dtw_filename, seq)
        print(f"DTW reference saved to: {dtw_filename}")
        break