import os

import matplotlib.pyplot as plt

import torch.optim as optim
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

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
    # update_dataset,
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


# method 1: randomly label 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(
                                        data_b, label_b,
                                                stratify=label_b[:, 0],
                                                test_size=0.2, random_state=42)



# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

a = y_train.reshape(-1)
# b = y_pool.reshape(-1)
c = y_test.reshape(-1)
# Define the PyTorch MLP model
device = 'cuda'
model = Autoencoder3d(is_reconst = False, is_classify = True)
model = model.to(device)
full_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch623_datalen50_accel.pth'
load_weights(
    full_model_path, model, my_device=device, is_dist=True, name_start_idx=0
    )

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Active learning loop
batch_size = 512
n_queries = 50
num_data, test_acc, train_loss = [], [], []  # 保存这两个数据为dict
for epoch in range(n_queries):
    # Train the model on the labeled data
    model.train()
    optimizer.zero_grad()

    # calculate batch of data
    predicts, labels = [], []
    if X_train.shape[0] < batch_size:
        loss, predict, label_vote = calculate_loss(
                                        X_train, y_train,
                                        device, model, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicts.append(predict.detach().cpu().numpy())
        labels.append(label_vote)

        num_data.append(X_train.shape[0])
        # Evaluate the model on the test set
        # eval_test(X_test, y_test, model, epoch)
        predict_labels, testacc = eval_test(X_test, y_test, model, epoch, criterion, device)
        test_acc.append(testacc)
        train_loss.append(loss.item())


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
        loss = criterion(probs, label_concat_vote)
        y_pred = torch.max(probs, dim=1).indices
        acc = accuracy_score(y_pred.detach().cpu().numpy(),
                             label_concat_vote.detach().cpu().numpy())
        num_data.append(X_train.shape[0])
        train_loss.append(np.average(losses))
        # print('The accuracy of training set at epoch %s is %f')
        print(f"Iteration {epoch + 1}, Train Accuracy: {acc:.4f}")

        # Evaluate the model on the test set
        # eval_test(X_test, y_test, model, epoch)
        predict_labels, testacc = eval_test(X_test, y_test, model, epoch, criterion, device)
        test_acc.append(testacc)

# plot
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
plt.title('Confusion Matrix of supervised learning at epoch %s'%str(epoch))
plt.show()

with open('supervise_results.pkl', 'wb') as f:
    results = {'num_data':num_data,
                 'test_acc': test_acc,
                 'train_loss': train_loss,
                 'y_true':label_concat_vote,
                 'y_pred':y_pred}
    pickle.dump(results, f)

with open('active_results.pkl', 'rb') as f:
    # results = {'num_data':num_data,
    #              'test_acc': test_acc,
    #              'train_loss': train_loss,
    #              'y_true':y_true,
    #              'y_pred':y_true}
    act_results = pickle.load(f)


plt.figure(figsize=(12,3))
plt.plot(np.array(results['train_loss']).astype(float), 'r', label='Supervised training loss')
plt.plot(np.array(act_results['train_loss']).astype(float), 'b', label='Active training loss')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Loss of iterations')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()


plt.figure(figsize=(12,3))
plt.plot(np.array(results['test_acc']).astype(float), 'r', label='Supervised learning')
sup_acc = np.array(results['test_acc']).astype(float)[-1]
# plt.plot([0,299],[sup_acc,sup_acc], color='red', linestyle='--', linewidth=1, label='Supervised learning')

plt.plot(np.array(act_results['test_acc']).astype(float), 'b', label='Active learning')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Average accuracy of test set')
# plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()

# plot number of data used for training todo
fig, axs = plt.subplots(2, 1, figsize=(12, 5))

# Plotting the training data
data_labels = np.arange(0, len(act_results['num_data']))
axs[0].bar(data_labels, act_results['num_data'], color='blue')
axs[0].set_title('Number of Training Samples - active learning')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Number of Samples')
# axs[0].set_ylim(0, max(train_data_counts) + 500)  # Set y-axis limit

# Plotting the test data
data_labels = np.arange(0, len(results['num_data']))
axs[1].bar(data_labels, results['num_data'], color='orange')
axs[1].set_title('Number of Training Samples - supervised learning')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Number of Samples')
# axs[1].set_ylim(0, max(test_data_counts) + 200)  # Set y-axis limit
# Adjust layout
plt.tight_layout()
# Show the plots
plt.show()
print("Pure supervised learning complete.")
