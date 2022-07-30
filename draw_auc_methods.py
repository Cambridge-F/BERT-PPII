import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


strict_mlp_fpr = np.load("FPRTPR/strict_methods/strict_mlp_fpr.npy")
strict_mlp_tpr = np.load("FPRTPR/strict_methods/strict_mlp_tpr.npy")

strict_svm_fpr = np.load("FPRTPR/strict_methods/strict_svm_fpr.npy")
strict_svm_tpr = np.load("FPRTPR/strict_methods/strict_svm_tpr.npy")


strict_rf_fpr = np.load("FPRTPR/strict_methods/strict_rf_fpr.npy")
strict_rf_tpr = np.load("FPRTPR/strict_methods/strict_rf_tpr.npy")


strict_knn_fpr = np.load("FPRTPR/strict_methods/strict_knn_fpr.npy")
strict_knn_tpr = np.load("FPRTPR/strict_methods/strict_knn_tpr.npy")

bert15_fpr = np.load("FPRTPR/strict_methods/bert15_fpr.npy")
bert15_tpr = np.load("FPRTPR/strict_methods/bert15_tpr.npy")

strict_mlp_roc_auc = metrics.auc(strict_mlp_fpr, strict_mlp_tpr)
strict_svm_roc_auc = metrics.auc(strict_svm_fpr, strict_svm_tpr)
strict_rf_roc_auc = metrics.auc(strict_rf_fpr, strict_rf_tpr)
strict_knn_roc_auc = metrics.auc(strict_knn_fpr, strict_knn_tpr)
bert15_roc_auc = metrics.auc(bert15_fpr, bert15_tpr)

print(strict_mlp_roc_auc)
print(strict_svm_roc_auc)
print(strict_rf_roc_auc)
print(strict_knn_roc_auc)
print(bert15_roc_auc)


plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(linestyle='--', label='No Skill')

plt.plot(strict_rf_fpr, strict_rf_tpr,label='ann(AUC= {:.3f})'.format(strict_mlp_roc_auc))
plt.plot(strict_svm_fpr, strict_svm_tpr,label='svm(AUC= {:.3f})'.format(strict_svm_roc_auc))
plt.plot(strict_rf_fpr, strict_rf_tpr,label='rf(AUC= {:.3f})'.format(strict_rf_roc_auc))
plt.plot(strict_knn_fpr, strict_knn_tpr,label='knn(AUC= {:.3f})'.format(strict_knn_roc_auc))
plt.plot(bert15_fpr,bert15_tpr,label='our(AUC= {:.3f})'.format(bert15_roc_auc))



plt.xlim([0.00, 1.0])
plt.ylim([0.00, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.legend(loc="lower right")
plt.savefig("pictures/strict_methods_auc.png")
