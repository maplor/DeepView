import os

import matplotlib.pyplot as plt

import torch.optim as optim
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from deepview.calculate_results.data.umineko.umineko_data import (
    read_umineko_path,
    extract_data_from_year_back,
)

from deepview.calculate_results.models.utils import (
    sliding_window,
    load_weights,
    # torch,
    majority_value,
    # np,
    # tqdm
    Autoencoder3d,
)
from deepview.calculate_results.viz.util_3_1 import (
    set_random_seed,
    label_dict,
    gaussian_std,
    calculate_loss,
    update_dataset,
    eval_test)


# Set a fixed random seed
seed_value = 2025
set_random_seed(seed_value)



raw_data, labeled_data = [], []
for year in ['2018', '2019', '2022']:
    dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
    data = np.load(dp % year, allow_pickle=True).item()
    # Access the individual components
    raw_ = data['raw_data']
    labeled_ = data['labeled_data']
    if year=='2018':
        df_raw_2018 = raw_
        df_2018 = labeled_
    elif year=='2019':
        df_raw_2019 = raw_
        df_2019 = labeled_
    elif year=='2022':
        df_raw_2022 = raw_
        df_2022 = labeled_
    else:
        print('Error: year not found')
        break

selected_df = pd.concat([df_raw_2018, df_raw_2019, df_raw_2022], ignore_index=True)
selected_df['label_id'] = selected_df['label'].map(label_dict)
selected_df['label_id'] = selected_df['label_id'].fillna(-2)

############################################
len_sw = 50
sensor_type = 'accel'
selected_columns = ['acc_x', 'acc_y', 'acc_z', 'label_id']
# 去掉-2的label
selected_df = selected_df[selected_df['label_id'] != -2]

selected_np = selected_df[selected_columns].values

data_np = selected_np[:, :-1]
tmp_b_stand = gaussian_std(data_np)

selected_np[:, :-1] = tmp_b_stand

tmp_b = sliding_window(selected_np, len_sw, len_sw)
# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]
##########################################

# label_concat_vote = majority_value(label_b)
# method 1: randomly label 80% training data and 20% test data
X_train_all, X_test, y_train_all, y_test = train_test_split(
    data_b, label_b,
    stratify=label_b[:, 0],  # no enough labels
    test_size=0.2, random_state=42)

# method 2 (active learning): randomly label 1% data for each class, add 1% for each iteration
full_list = list(range(len(y_train_all)))  # +1 to include n itself
one_percent_count = max(1, int(len(full_list) * 0.01))
# todo: check number of classes, add 1% of every class
random_one_percent = np.random.choice(full_list, size=one_percent_count, replace=False)
train_indices = random_one_percent
pool_indices = list(set(full_list) - set(train_indices))

X_train = X_train_all[train_indices]
y_train = y_train_all[train_indices]
X_pool = X_train_all[pool_indices]
y_pool = y_train_all[pool_indices]


# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_pool = torch.tensor(X_pool, dtype=torch.float32)
y_pool = torch.tensor(y_pool, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

a = y_train.reshape(-1)
b = y_pool.reshape(-1)
c = y_test.reshape(-1)
# Define the PyTorch MLP model
device = 'cuda'
model = Autoencoder3d(is_reconst=False, is_classify=True)
model = model.to(device)
full_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch623_datalen50_accel.pth'
load_weights(
    full_model_path, model, my_device=device, is_dist=True, name_start_idx=0
)

learning_rate = 0.0001
# Active learning loop
batch_size = 512
n_queries = 100
iteration = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=50)

X_unlabeled = X_pool.to(device)  # all-train-test
y_unlabeled = y_pool.to(device)

num_data, test_acc, train_loss = [], [], []  # 保存这两个数据为dict
for epoch in range(n_queries):
    # Train the model on the labeled data
    model.train()
    optimizer.zero_grad()

    for repeat in range(iteration):
        # calculate batch of data
        predicts, labels = [], []
        if X_train.shape[0] < batch_size:
            loss, predict, label_vote = calculate_loss(
                X_train, y_train,
                device, model,
             criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            label_concat_vote = majority_value(y_train)
            label_concat_vote = torch.asarray(label_concat_vote).to(device=device, dtype=torch.long)
            # loss = criterion(predict, label_concat_vote)
            y_pred = torch.max(predict, dim=1).indices
            acc = accuracy_score(y_pred.detach().cpu().numpy(),
                                 label_concat_vote.detach().cpu().numpy())
            num_data.append(X_train.shape[0])
            train_loss.append(loss.item())
            # print('The accuracy of training set at epoch %s is %f')
            print(f"Iteration {epoch + 1}, Train Accuracy: {acc:.4f}")

            # Predict on the unlabeled data
            if repeat == iteration - 1:
                (X_train, y_train,
                 X_unlabeled, y_unlabeled) = update_dataset(
                    model,
                    batch_size,
                    one_percent_count,
                    X_unlabeled,
                    y_unlabeled,
                    X_train,
                    y_train,
                           criterion,
                           device)

                # Evaluate the model on the test set
                predict_labels, testacc = eval_test(X_test, y_test, model, epoch, criterion, device)
                test_acc.append(testacc)

        else:
            model.train()
            optimizer.zero_grad()
            # Normal batch processing
            num_batches = (X_train.shape[0]) // batch_size  # Calculate the number of batches
            probs = []
            losses = []
            for i in range(num_batches):
                # Calculate the start and end index for the current batch
                start_index = i * batch_size
                end_index = min(start_index + batch_size, X_train.shape[0])  # Ensure we don't go out of bounds

                # Get the batch from A and B
                batch_x = X_train[start_index:end_index]  # Shape will be (size of batch, 3, 100)
                batch_y = y_train[start_index:end_index]  # Shape will be (size of batch,)
                loss, prob, label_vote = calculate_loss(
                    batch_x, batch_y, device, model, criterion)
                probs.append(prob)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_x = X_train[end_index:]  # Shape will be (size of batch, 3, 100)
            batch_y = y_train[end_index:]  # Shape will be (size of batch,)
            loss, prob, label_vote = calculate_loss(
                batch_x, batch_y, device, model, criterion)
            probs.append(prob)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.concat(probs).to(device)
            label_concat_vote = majority_value(y_train)
            label_concat_vote = torch.asarray(label_concat_vote).to(device=device, dtype=torch.long)
            # loss = criterion(probs, label_concat_vote)
            y_pred = torch.max(probs, dim=1).indices
            acc = accuracy_score(y_pred.detach().cpu().numpy(),
                                 label_concat_vote.detach().cpu().numpy())
            num_data.append(X_train.shape[0])
            # if losses > 0.3:
            #     print('')
            train_loss.append(np.average(losses))
            # print('The accuracy of training set at epoch %s is %f')
            print(f"Iteration {epoch + 1}, Train Accuracy: {acc:.4f}")

            # Predict on the unlabeled data
            if repeat == iteration - 1:
                if X_unlabeled.shape[0] > 0:
                    (X_train, y_train,
                     X_unlabeled, y_unlabeled) = update_dataset(
                        model,
                        batch_size,
                        one_percent_count,
                        X_unlabeled,
                        y_unlabeled,
                        X_train,
                        y_train,
                           criterion,
                           device)
                # else:
                #     print('')

                # Evaluate the model on the test set
                predict_labels, testacc = eval_test(X_test, y_test, model, epoch, criterion, device)
                test_acc.append(testacc)

    for param_group in optimizer.param_groups:
        print("Learning Rate:", param_group['lr'])


# Example ground truth labels and predicted labels
y_true = y_test.detach().cpu().numpy()  # Ground truth labels
y_pred = predict_labels.detach().cpu().numpy()  # Predicted labels
label_concat_vote = majority_value(y_true)
# Compute confusion matrix
cm = confusion_matrix(label_concat_vote, y_pred)

# Create a confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of active learning at epoch %s' % str(epoch))
plt.show()


with open('active_results.pkl', 'wb') as f:
    results = {'num_data': num_data,
               'test_acc': test_acc,
               'train_loss': train_loss,
               'y_true': label_concat_vote,
               'y_pred': y_pred}
    pickle.dump(results, f)

print("Active learning complete.")
