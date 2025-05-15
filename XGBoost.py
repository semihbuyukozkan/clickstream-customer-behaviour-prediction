import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("dataset/features.csv")
X = df.drop(columns=['session_id', 'purchase', 'last_event_type'])
y = df['purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 4. Modeli eğit
model.fit(X_train, y_train)

# 5. Tahmin ve olasılıklar
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4, output_dict=True)

# Manuel metrikler (ekstra kontrol için)
roc_auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC AUC Score:", roc_auc)

results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'Score': [accuracy, precision, recall, f1, roc_auc]
})
results_df.to_csv("results/xgboost_results.csv", index=False)

pd.DataFrame({'y_true': y_test, 'y_prob': y_prob}).to_csv("results/xgboost_curve_data.csv", index=False)

# Feature importance grafiği
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.title("XGBoost Modeli Özellik Önemi Grafiği", fontsize=13)
plt.xlabel("Önem Düzeyi", fontsize=11)
plt.tight_layout()
plt.grid(alpha=0.3, axis='x')
plt.savefig("plots/feature_importance_xgb.png")
plt.close()
