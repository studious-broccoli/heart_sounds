import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import torch
import matplotlib
matplotlib.use("Agg")


def basic_cm_snsp(y_true, y_pred, filename, class_names=None):

    # Get unique class names (sorted for consistency)
    if class_names is None:
        class_names = sorted(set(y_true).union(set(y_pred)))

    class_names = sorted(set(y_true).union(set(y_pred)))
    num_classes = len(class_names)

    fig = plt.figure(f"Confusion Matrix", figsize=(10, 10))
    np.seterr(invalid='ignore')

    cf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)

    try:
        f1 = f1_score(y_true, y_pred)
    except:
        label_to_int = {'ABNORMAL': 0, 'NORMAL': 1}
        y_true_int = [label_to_int[label] for label in y_true]
        y_pred_int = [label_to_int[label] for label in y_pred]
        f1 = f1_score(y_true_int, y_pred_int)

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


def visualize_feature_maps(model, input_tensor, filename):
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))

    with torch.no_grad():
        _ = model(input_tensor)

    for layer_name in activation:
        act = activation[layer_name][0]  # first sample
        num_filters = act.shape[0]
        fig, axes = plt.subplots(1, min(num_filters, 6), figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(act[i].cpu(), cmap='viridis', aspect='auto')
            ax.set_title(f"{layer_name} | Filter {i}")
        plt.tight_layout()
        plt.savefig(filename.replace(".png", f"_{layer_name}.png"))
        plt.close()


def visualize_saliency(model, input_tensor, label_idx, filename):
    input_tensor.requires_grad_()
    model.eval()

    output = model(input_tensor)
    loss = output[0, label_idx]
    loss.backward()

    # Get saliency (absolute value of gradient)
    saliency = input_tensor.grad.data.abs().cpu().numpy()

    # Shape sanity check
    if saliency.ndim == 4:
        saliency = saliency[0, 0]  # shape: (13, 300)
    elif saliency.ndim == 3:
        saliency = saliency[0]     # shape: (13, 300)

    plt.figure(figsize=(10, 4))
    plt.imshow(saliency, cmap='hot', aspect='auto')
    plt.title("Saliency Map (input gradients)")
    plt.colorbar()
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_losses(losses, filename="loss.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, marker='o', label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

