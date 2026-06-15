"""
Training & evaluation module
============================
Reproduces the training, freezing strategy, evaluation and active-learning
sample-selection helpers of the notebook.

Two-headed training per active-learning iteration:
  stage 1 (iteration < warmup): supervised-contrastive training of encoders.
  stage 2: freeze encoders, train classifier only (50 epochs).
  stage 3: unfreeze everything, train classifier head (50 epochs).

Evaluation:
  * evaluate_model            -> plain supervised accuracy / macro / micro F1.
  * evaluate_supContrast_model-> same but with a confidence threshold that
                                 relabels low-confidence predictions to -1.

Active learning:
  * uncertainty_sampling      -> entropy-based selection of `update_size`
                                 most-uncertain unlabelled windows.
"""

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score

from data_preprocessing import majority_value


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_model(model, loader, criterion, optimizer, epochs=500,
                device="cpu", if_contrast=True):
    model.train()
    avg_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels = batch
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)

            data = data.to(device=device, dtype=torch.float)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            outputs, features = model(data, if_contrast)
            loss = criterion(outputs, label_vote)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss.append(np.mean(losses))
    return model, avg_loss


# --------------------------------------------------------------------------- #
# freeze / unfreeze strategies
# --------------------------------------------------------------------------- #
def freeze_encoders(model):
    for param in model.acc_encoder.parameters():
        param.requires_grad = False
    for param in model.pre_encoder.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_encoders(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(model, loader, device="cpu"):
    model.eval()
    ground_truth, prediction = [], []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float)
            label_vote = torch.from_numpy(majority_value(labels)).to(
                device=device, dtype=torch.long)
            predicts, _ = model(data, if_contrast=False)
            _, predictions = torch.max(predicts, dim=1)
            ground_truth.append(label_vote.detach().cpu().numpy())
            prediction.append(predictions.detach().cpu().numpy())

    gt = np.concatenate(ground_truth, axis=0)
    pr = np.concatenate(prediction, axis=0)
    accuracy = accuracy_score(gt, pr)
    macro_f1 = f1_score(gt, pr, average="macro")
    micro_f1 = f1_score(gt, pr, average="micro")
    print(f"Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, "
          f"Micro F1 Score: {micro_f1:.2f}")
    return accuracy, macro_f1, micro_f1, prediction, ground_truth


def evaluate_supContrast_model(model, loader, cal_threshold=0.0, device="cpu"):
    model.eval()
    data_list, predlabel_list, truelabel_list = [], [], []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float)
            label_vote = torch.from_numpy(majority_value(labels)).to(
                device=device, dtype=torch.long)
            predicts, _ = model(data, if_contrast=False)
            # classifier emits logits -> softmax for a real confidence score
            predicts = torch.softmax(predicts, dim=1)
            max_values, predictions = torch.max(predicts, dim=1)

            # relabel low-confidence predictions to -1
            predictions[max_values < cal_threshold] = -1

            pos_positions = torch.nonzero(predictions != -1).squeeze()
            if pos_positions.numel() > 0:
                data_list.append(data[pos_positions].detach().cpu().numpy())
                predlabel_list.append(
                    predictions[pos_positions].detach().cpu().numpy())
                truelabel_list.append(
                    label_vote[pos_positions].detach().cpu().numpy())
            else:
                data_list.append(data.detach().cpu().numpy())
                predlabel_list.append(predictions.detach().cpu().numpy())
                truelabel_list.append(label_vote.detach().cpu().numpy())

    concat_data = np.concatenate(data_list, axis=0)
    concat_label = np.concatenate(predlabel_list, axis=0)
    true_label = np.concatenate(truelabel_list, axis=0)
    accuracy = accuracy_score(true_label, concat_label)
    macro_f1 = f1_score(true_label, concat_label, average="macro")
    micro_f1 = f1_score(true_label, concat_label, average="micro")
    print(f"Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, "
          f"Micro F1 Score: {micro_f1:.2f}")
    return (accuracy, macro_f1, micro_f1,
            concat_data, concat_label, true_label)


def AE_eval_time_series(loader, model, device="cpu"):
    """Run the model over a loader, collecting 32-d shared features, raw
    samples, softmax predictions and labels (used for the visualisations)."""
    model.eval()
    representation_list, sample_list, label_list, pred_list = [], [], [], []
    with torch.no_grad():
        for sample, label in loader:
            sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
            output, x_encoded = model(sample, if_contrast=False)
            representation_list.append(x_encoded.detach().cpu().numpy())
            sample_list.append(sample.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            pred_list.append(output.detach().cpu().numpy())
    return representation_list, sample_list, pred_list, label_list


# --------------------------------------------------------------------------- #
# active learning: entropy-based uncertainty sampling
# --------------------------------------------------------------------------- #
def uncertainty_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                         model, update_size, device="cpu"):
    model.eval()
    with torch.no_grad():
        # batch the forward pass so a large unlabelled pool (e.g. 2 s windows ->
        # ~190k samples) does not blow up memory.
        chunks = []
        for i in range(0, len(X_unlabeled), 8192):
            xb = torch.tensor(X_unlabeled[i:i + 8192], device=device,
                              dtype=torch.float32)
            _, emb = model(xb, if_contrast=True)
            logits = model.classifier(emb)
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        probs = np.concatenate(chunks, axis=0)
        uncertainty = entropy(probs.T)

        selected_indices = np.argsort(uncertainty)[-update_size:]
        selected_samples = X_unlabeled[selected_indices]
        selected_labels = y_unlabeled[selected_indices]

        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        print(f"Labeled samples: {len(X_labeled)}")
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
