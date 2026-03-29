import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Added for better visualization
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("Crop_recommendation.csv")

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

# =========================
# 6. Apply SMOTE & Capture States
# =========================

# Capture "Before" state
before_smote_counts = y_train.value_counts().sort_index()

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Capture "After" state
after_smote_counts = pd.Series(y_train_resampled).value_counts().sort_index()

# Update training variables to the resampled versions
X_train, y_train = X_train_resampled, y_train_resampled

# =========================
# 7. Feature Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 8. Machine Learning Models
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

    results.append([
        name, 
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions, average="weighted"),
        recall_score(y_test, predictions, average="weighted"),
        f1_score(y_test, predictions, average="weighted")
    ])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

# =========================
# 10. Visualizations (Updated)
# =========================

# --- NEW: SMOTE Comparison Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

before_smote_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title("Class Distribution BEFORE SMOTE")
ax1.set_xlabel("Crop Label ID")
ax1.set_ylabel("Count")

after_smote_counts.plot(kind='bar', ax=ax2, color='salmon')
ax2.set_title("Class Distribution AFTER SMOTE")
ax2.set_xlabel("Crop Label ID")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.show()

# --- Model Accuracy Comparison ---
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

# =========================
# 11-14. Final Processing (Confusion Matrix & Saving)
# =========================
best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

# Final train on resampled or full data as per your preference
# (Using resampled X_train/y_train here to ensure balance in final model)
if best_model_name in ["Logistic Regression", "KNN", "SVM"]:
    best_model.fit(X_train_scaled, y_train)
else:
    best_model.fit(X_train, y_train)

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print(f"\nPipeline complete. Best Model: {best_model_name}")