import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df_weight = pd.read_csv("results/logreg_weight_results.csv")
fpr_w, tpr_w, _ = roc_curve(df_weight['y_true'], df_weight['y_prob'])
auc_w = auc(fpr_w, tpr_w)

plt.figure()
plt.plot(fpr_w, tpr_w, label=f'Logistic Regression (Weight) ROC curve (AUC = {auc_w:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Weight)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("plots/roc_logreg_weight.png")
plt.show()
