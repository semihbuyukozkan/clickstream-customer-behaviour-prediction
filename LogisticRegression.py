#SKLEARN BALANCED İLE DENGELENMİŞ VERİ
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("dataset/features.csv")

X = df.drop(columns=['session_id', 'purchase', 'last_event_type'])
y = df['purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

pd.DataFrame({'y_true': y_test, 'y_prob': y_prob}).to_csv("results/logreg_weight_results.csv", index=False)

results_df = pd.DataFrame([{
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_prob)
}])
results_df.to_csv("results/logreg_weight_metrics.csv", index=False)
