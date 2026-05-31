import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)

if __package__:
    from .label_utils import generate_class_labels_for_vis, return_species_jp_name
else:
    from label_utils import generate_class_labels_for_vis, return_species_jp_name


logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_gt, y_pred, cfg, figsize=(9, 7)):
    species_jp_name = return_species_jp_name(cfg)
    class_labels = generate_class_labels_for_vis(species_jp_name)

    labels_int = np.arange(0, len(class_labels), 1).tolist()

    cm = confusion_matrix(y_gt, y_pred, labels=labels_int)

    df_cm = pd.DataFrame(data=cm, index=class_labels, columns=class_labels)
    logger.debug("Confusion matrix:\n%s", df_cm)

    fig = plt.figure(figsize=figsize)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_precision_scores = (df_cm / np.sum(df_cm)).values.flatten()
    group_percentages = ["{0:.2f}".format(value)
                         for value in group_precision_scores]
    annot_labels = [
        f"{v1}\n({v2})" for v1,
        v2 in zip(
            group_counts,
            group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(len(labels_int), len(labels_int))
    ax = sns.heatmap(
        df_cm / np.sum(df_cm),
        # df_cm,
        vmin=0, vmax=1.0,
        square=True, cbar=True, annot=annot_labels, fmt='',
        cmap='Blues'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Prediction", fontsize=14, rotation=0, labelpad=10)
    plt.ylabel("Ground Truth", fontsize=14, labelpad=10)
    ax.set_ylim(len(cm), 0)
    fig.tight_layout()
    plt.show()
    plt.close()

    fig_cm = fig

    return cm, df_cm, fig_cm


def plot_window_ax(ax, X, label, npz_file_name):
    '''
    X: numpy array
    '''
    acc_x = X[0].transpose(1, 0)[0]
    acc_y = X[0].transpose(1, 0)[1]
    acc_z = X[0].transpose(1, 0)[2]
    window_size = len(X[0])
    data_number = list(range(1, window_size + 1, 1))
    # color_list = ['#EE6677', '#228833', '#4477AA']
    color_list = ['#D81B60', '#FFC107', '#1E88E5']
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_x, label="x", color=color_list[0])
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_y, label="y", color=color_list[1])
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_z, label="z", color=color_list[2])
    if npz_file_name is None:
        logger.debug("No title")
    else:
        if label is None:
            ax.set_title(f"{npz_file_name}", pad=10)
        else:
            ax.set_title(f"{npz_file_name} | label_id: {(int(label))}", pad=10)
    ax.set_xlabel("t")
    ax.set_ylabel("g")
    ax.set_xticks(np.arange(0, 51, 10), fontsize=18)
    ax.set_yticks(np.arange(-4.0, 4.2, 2.0))
    xticklabels = ['{:,.0f}'.format(x) for x in np.arange(0, 51, 10.0)]
    yticklabels = ['{:,.1f}'.format(x) for x in np.arange(-4.0, 4.1, 2.0)]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(-3, 53)
    ax.set_ylim(-4.5, 4.5)
    ax.legend(ncol=3)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
    ax.grid(axis='both', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', alpha=0.5)
    # plt.show()
    # plt.close()

    return ax


def generate_test_score_df(cfg, y_gt, y_pred, test_animal_id):
    model_name = cfg.model.model_name
    species_jp_name = return_species_jp_name(cfg)
    class_labels = generate_class_labels_for_vis(species_jp_name)
    labels_int = np.arange(0, len(class_labels), 1).tolist()

    category = ["Macro", "Weighted"]
    category.extend(class_labels)

    model = np.array([model_name] * len(category))
    test_id = np.array([test_animal_id] * len(category))

    species = np.array([cfg.dataset.species] * len(category))

    # macro average
    precision_macro = precision_score(y_gt, y_pred, average="macro")
    recall_macro = recall_score(y_gt, y_pred, average="macro")
    f1_macro = f1_score(y_gt, y_pred, average="macro")
    IoU_macro = jaccard_score(y_gt, y_pred, average="macro")

    # weighted average
    precision_weighted = precision_score(y_gt, y_pred, average="weighted")
    recall_weighted = recall_score(y_gt, y_pred, average="weighted")
    f1_weighted = f1_score(y_gt, y_pred, average="weighted")
    IoU_weighted = jaccard_score(y_gt, y_pred, average="weighted")

    # scores for each class
    precision_scores = precision_score(y_gt, y_pred, average=None, labels=labels_int)
    recall_scores = recall_score(y_gt, y_pred, average=None, labels=labels_int)
    f1_scores = f1_score(y_gt, y_pred, average=None, labels=labels_int)
    IoU_scores = jaccard_score(y_gt, y_pred, average=None, labels=labels_int)

    # df for storing results
    data_dict = {
        'Model': model,
        'Category': category,
        'Test_ID': test_id,
        'Species': species,
        'Precision': np.append(
            np.append(
                precision_macro,
                precision_weighted),
            precision_scores),
        'Recall': np.append(
            np.append(
                recall_macro,
                recall_weighted),
            recall_scores),
        'F1': np.append(
            np.append(
                f1_macro,
                f1_weighted),
            f1_scores),
        'IoU': np.append(
            np.append(
                IoU_macro,
                IoU_weighted),
            IoU_scores),
    }

    df = pd.DataFrame(data=data_dict)

    return df