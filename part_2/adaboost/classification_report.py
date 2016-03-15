import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


def classification_report(y_true, y_pred):
    ''' Computes classification metrics
    :param y_true - original class label
    :param y_pred - predicted class label
    :return macro_f1_measure
    '''

    labels = unique_labels(y_true, y_pred)

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None)

    f1_macro = 0
    precision_macro = 0
    recall_macro = 0

    for i, label in enumerate(labels):
        f1_macro += f1[i]
        precision_macro += p[i]
        recall_macro += r[i]

    macro_f1_measure = f1_macro / labels.size
    return macro_f1_measure
