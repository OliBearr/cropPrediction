# Agricultural Dataset Analysis Using Machine Learning
# Crop Recommendation Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("Crop_recommendation.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# =========================
# 2. Check Missing Values
# =========================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 3. Encode Target Variable
# =========================

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# =========================
# 4. Feature Selection
# =========================

X = df.drop(["label", "label_encoded"], axis=1)
y = df["label_encoded"]

# =========================
# 5. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# =========================
# 6. Feature Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. Machine Learning Models
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    results.append([name, accuracy, precision, recall, f1])

# =========================
# 8. Results Table
# =========================

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print("\nModel Comparison Results:")
print(results_df)

# =========================
# 9. Visualization
# =========================

# Accuracy Comparison Chart
plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# Temperature Distribution
plt.figure()
plt.hist(df["temperature"], bins=20)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =========================
# 10. Confusion Matrix
# =========================

best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# Correlation Heatmap
# =========================

import matplotlib.pyplot as plt

correlation = df.corr(numeric_only=True)

plt.figure()

plt.imshow(correlation)

plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

import joblib

# Get best model based on accuracy
best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

# IMPORTANT: Retrain best model on FULL dataset
if best_model_name in ["Logistic Regression", "KNN", "SVM"]:
    best_model.fit(scaler.fit_transform(X), y)
else:
    best_model.fit(X, y)

# Save files
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model, scaler, and encoder saved!")