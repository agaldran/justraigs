from sklearn import metrics
import math
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import average_precision_score
import numpy as np
def get_imb_metrics(y_true, y_prob, thresh=0.5):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    sens, spec = tpr, 1 - fpr
    sens_at_95_spec = sens[spec > 0.95][-1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    au_prc = auc(recall, precision)
    ap = average_precision_score(y_true, y_prob)
    # p_auc = roc_auc_score(y_true, y_prob, max_fpr=0.10)  # partial AUC (90-100% specificity)
    # bacc = balanced_accuracy_score(y_true, y_prob > thresh)
    # f1 = f1_score(y_true, y_prob)
    # return au_prc, sens_at_95_spec, p_auc, bacc, f1

    return au_prc, ap, sens_at_95_spec


def hamming_loss(true_labels, prob_labels, thresh=0.5):
    predicted_labels = prob_labels > thresh
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # keep only features for which there was agreement, see data_splitting_features.ipynb
    y_true = true_labels[(true_labels <= 0.1) | (true_labels >= 0.9)]
    y_true = y_true > 0.5  # need to binarize potentially soft labels now
    y_pred = predicted_labels[(true_labels <= 0.1) | (true_labels >= 0.9)]
    # Calculate the hamming distance that is basically the total number of mismatches
    hamming_distance = np.sum(np.not_equal(y_true, y_pred))

    # Calculate the total number of labels
    total_corrected_labels = y_true.size

    # Compute the Modified Hamming loss
    loss = hamming_distance / total_corrected_labels
    return loss