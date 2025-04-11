import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


def basic_cm_snsp(y_true, y_pred, filename, class_names=None):

    # Get unique class names (sorted for consistency)
    if class_names is None:
        class_names = sorted(set(y_true).union(set(y_pred)))

    class_names = sorted(set(y_true).union(set(y_pred)))
    num_classes = len(class_names)

    fig = plt.figure(f"Confusion Matrix", figsize=(10, 10))
    np.seterr(invalid='ignore')

    cf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)

    f1 = f1_score(y_true, y_pred)

    total_preds = np.sum(cf_matrix)
    tn = total_preds - cf_matrix.sum(axis=1)
    tp = [cf_matrix[i, i] for i in range(0, num_classes)]
    fp = cf_matrix.sum(axis=0) - tp
    specificity = tn / (tn + fp)

    # compute recall (sensitivity) for each class
    # R = TP / (TP + FN)
    denom = cf_matrix.sum(axis=1)[:, None]
    recall = cf_matrix / denom if denom.any() > 0. else 1.

    # only include precision and recall labels along matrix diagnoal
    diag_indx = [v * (num_classes + 1) for v in range(0, num_classes)]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_recall = ["" if (i not in diag_indx or np.isnan(v)) else "Se: {0:.1%}".format(v) for i, v in enumerate(recall.flatten())]
    group_specificity = ["" if (i not in diag_indx) else "Sp: {0:.1%}".format(specificity[i % num_classes]) for i in range(0, num_classes * num_classes)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_counts, group_recall, group_specificity)]
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names, rotation=45)

    title = f"Abnormal Heart Sound Detection\nF1-Score: {f1 * 100:.2f}% (N = {len(y_true)})"
    plt.title(title)
    plt.tight_layout()

    # Display the visualization of the Confusion Matrix.
    print(f"saving Confusion Matrix to {filename}")
    plt.savefig(filename)
    plt.close('all')