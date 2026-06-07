import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("creditcard.csv")
print(f"Shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} / {len(df)} ({df['Class'].mean()*100:.3f}%)")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"]   = scaler.fit_transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

# ── 3. Train-test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ── 4. SMOTE to fix class imbalance ──────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Fraud: {y_train_sm.sum()}, Legit: {(y_train_sm==0).sum()}")

# ── 5. Train Logistic Regression ─────────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sm, y_train_sm)
lr_pred = lr.predict(X_test)
print("── Logistic Regression ──")
print(classification_report(y_test, lr_pred, target_names=["Legit", "Fraud"]))
print(f"ROC-AUC: {roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]):.4f}")

# ── 6. Train Random Forest ───────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
rf_pred = rf.predict(X_test)
print("── Random Forest ──")
print(classification_report(y_test, rf_pred, target_names=["Legit", "Fraud"]))
roc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc:.4f}")

# ── 7. Confusion matrix plot ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit","Fraud"],
            yticklabels=["Legit","Fraud"])
plt.title("Random Forest - Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion_matrix.png")

# ── 8. Feature importance plot ────────────────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.nlargest(10).sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved feature_importance.png")

# ── 9. Save model & scaler ────────────────────────────────────────────────────
joblib.dump(rf, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print("\nModel saved as fraud_model.pkl")
print("Run app.py next to launch the UI!")