"""
Visualisation module
=====================
Reproduces the post-training visualisation cells of the notebook:

  1. Build a "combined" set = labelled windows + a raw time slice
     [startloc:endloc] of the recording, run it through the trained model,
     and UMAP-project the 32-d features to 2-D.
  2. Label propagation (sklearn LabelSpreading, knn kernel) from the few known
     labels over the UMAP embedding.
  3. Plotly scatter HTMLs: predicted-label, propagated-label, true-label and
     "before propagation" views.
  4. Matplotlib time-series PDFs: ground-truth vs propagated labels,
     ground-truth vs model-prediction labels, and the raw acceleration trace.

All file names mirror the notebook's outputs; everything is written into
`out_dir`.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from umap import UMAP
import plotly.express as px

from data_preprocessing import (
    majority_value,
    data_loader_umineko,
    labeldict_findstr,
    labeldict_wounknown_findstr,
    label_colors,
    act_color_dict,
)
from train_eval import AE_eval_time_series
from torch.utils.data import DataLoader


def run_visualizations(model, data_b, vote_label, X_labeled, y_labeled,
                       out_dir, startloc=134000, endloc=139000,
                       max_iter=20, warmup=20, batch_size=4000, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)

    # ----- 1. combined set + features + UMAP ----------------------------- #
    combinedata = np.concatenate([X_labeled, data_b[startloc:endloc]])
    combinelabel_true = np.concatenate([y_labeled, vote_label[startloc:endloc]])

    truelabeled_dataset = data_loader_umineko(
        combinedata.astype(float), combinelabel_true.astype(int))
    truelabeled_loader = DataLoader(truelabeled_dataset, batch_size=batch_size,
                                    shuffle=False, drop_last=False)
    repres_list, _, pred_list, combinelabel_list = AE_eval_time_series(
        truelabeled_loader, model, device)

    repre_concat = np.concatenate(repres_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)
    proj_3d_gyro = UMAP(n_components=2).fit_transform(repre_reshape)

    # model-prediction labels for the combined set
    combinepred = np.concatenate(pred_list)
    predictions = np.argmax(combinepred, axis=1)
    all_labels = np.concatenate(combinelabel_list)  # ground-truth (voted)

    # ----- 2. label propagation ----------------------------------------- #
    # mark the appended raw slice as unlabelled (-1)
    combineunlabel = np.concatenate(
        [y_labeled, np.full(vote_label[startloc:endloc].shape, -1)])
    unlabeled_dataset = data_loader_umineko(
        combinedata.astype(float), combineunlabel.astype(int))
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False)
    _, _, _, unlabel_list = AE_eval_time_series(unlabeled_loader, model, device)
    all_labels_before = np.concatenate(unlabel_list)

    scaler = StandardScaler()
    data_umap = scaler.fit_transform(proj_3d_gyro)

    label_spread = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2)
    label_spread.fit(data_umap, all_labels_before)
    propagated_labels = label_spread.transduction_

    # ----- 3. plotly scatter HTMLs --------------------------------------- #
    _plot_before_propagation(data_umap, all_labels_before, out_dir,
                             max_iter, warmup)
    _plot_simple_scatter(
        data_umap, [labeldict_findstr[i] for i in propagated_labels],
        os.path.join(out_dir,
                     f"label_propagation_predlabel_iter{max_iter}_warmup{warmup}.html"))
    _plot_simple_scatter(
        data_umap, [labeldict_findstr[i] for i in all_labels],
        os.path.join(out_dir,
                     f"true_label_all_iter{max_iter}_warmup{warmup}.html"))

    # ----- 4. matplotlib time-series PDFs -------------------------------- #
    new_data_len = len(data_b[startloc:endloc])
    true_newlabel = all_labels[-new_data_len:]
    prop_newlabel = propagated_labels[-new_data_len:]
    pred_newlabel = predictions[-new_data_len:]

    _plot_label_timeseries(
        true_newlabel, prop_newlabel, new_data_len,
        "Groundtruth (lines) vs Propagated labels (rectangles)",
        os.path.join(out_dir, f"propagationlabel_{max_iter}_warmup{warmup}.pdf"))
    _plot_label_timeseries(
        true_newlabel, pred_newlabel, new_data_len,
        "Groundtruth (lines) vs Model-prediction labels (rectangles)",
        os.path.join(out_dir, f"modelpred_accel_{max_iter}_warmup{warmup}.pdf"))
    _plot_raw_accel(data_b, startloc, endloc,
                    os.path.join(out_dir, f"propagation_accel_{max_iter}.pdf"))

    return {
        "proj_3d_gyro": proj_3d_gyro,
        "data_umap": data_umap,
        "propagated_labels": propagated_labels,
        "predictions": predictions,
        "true_labels": all_labels,
    }


# --------------------------------------------------------------------------- #
# plotly helpers
# --------------------------------------------------------------------------- #
def _style(fig):
    fig.update_traces(marker=dict(size=6, opacity=0.5, line=dict(width=0)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="black")),
        xaxis=dict(title="UMAP 0", title_font=dict(color="black"),
                   tickfont=dict(color="black")),
        yaxis=dict(title="UMAP 1", title_font=dict(color="black"),
                   tickfont=dict(color="black")),
        title=dict(font=dict(color="black")),
    )
    return fig


def _plot_simple_scatter(data_umap, color_str, path):
    fig = px.scatter(data_umap, x=0, y=1, color=color_str,
                     labels={"activity": "activity"},
                     color_discrete_map=act_color_dict)
    _style(fig).write_html(path)


def _plot_before_propagation(data_umap, all_labels_before, out_dir,
                             max_iter, warmup):
    valid = all_labels_before != -1
    unknown = all_labels_before == -1
    valid_str = [labeldict_findstr[i] for i in all_labels_before[valid]]
    fig = px.scatter(data_umap[valid], x=0, y=1, color=valid_str,
                     labels={"activity": "activity"},
                     color_discrete_map=act_color_dict)
    fig.add_scatter(
        x=data_umap[unknown][:, 0], y=data_umap[unknown][:, 1], mode="markers",
        marker=dict(size=6, opacity=0.2, color="lightgrey", line=dict(width=0)),
        name="unknown")
    _style(fig).write_html(
        os.path.join(out_dir,
                     f"label_propagation_before_iter{max_iter}_warmup{warmup}.html"))


# --------------------------------------------------------------------------- #
# matplotlib helpers
# --------------------------------------------------------------------------- #
def _plot_label_timeseries(true_newlabel, overlay_label, new_data_len,
                           title, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, label_name in labeldict_findstr.items():
        if label >= 0:
            idx = np.where(true_newlabel == label)[0]
            ax.scatter(idx, [label] * len(idx), s=1,
                       color=label_colors[label], label=label_name)
    for start in range(0, new_data_len, 1):
        for label in np.unique(overlay_label[start:start + 1]):
            if (label >= 0) and (label in label_colors):
                ax.add_patch(patches.Rectangle(
                    (start, label - 0.3), 1, 0.6, linewidth=0,
                    edgecolor=None, facecolor=label_colors[label], alpha=0.2))
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Activity")
    ax.set_title(title)
    ax.set_yticks(list(labeldict_wounknown_findstr.keys()))
    ax.set_yticklabels(list(labeldict_wounknown_findstr.values()))
    ax.legend(markerscale=5, fontsize=8, loc="upper right")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_accel(data_b, startloc, endloc, path):
    plt.figure(figsize=(12, 4))
    acc_data = np.concatenate(
        data_b[startloc:endloc, :3, :].transpose(0, 2, 1), axis=0)
    plt.plot(acc_data[:, 0], label="axis_X")
    plt.plot(acc_data[:, 1], label="axis_Y")
    plt.plot(acc_data[:, 2], label="axis_Z")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration Value [G]")
    plt.title("Raw Acceleration Data")
    plt.legend(markerscale=5, fontsize=12, loc="upper right")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
