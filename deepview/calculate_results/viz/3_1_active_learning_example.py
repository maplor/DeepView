import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.8, random_state=42)
X_test, X_pool, y_test, y_pool = train_test_split(X_pool, y_pool, test_size=0.5, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_pool = torch.tensor(X_pool, dtype=torch.float32)
y_pool = torch.tensor(y_pool, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initial labeled dataset (small)
n_initial = 10
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_labeled = X_train[initial_idx]
y_labeled = y_train[initial_idx]

# Remove initial samples from the unlabeled pool
X_unlabeled = torch.cat([X_train[:initial_idx[0]], X_train[initial_idx[-1] + 1:]])
y_unlabeled = torch.cat([y_train[:initial_idx[0]], y_train[initial_idx[-1] + 1:]])


# Define the PyTorch MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


# Initialize the model
input_size = X.shape[1]
hidden_size = 100
output_size = len(np.unique(y))
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Active learning loop
n_queries = 20
for i in range(n_queries):
    # Train the model on the labeled data
    model.train()
    optimizer.zero_grad()
    outputs = model(X_labeled)  # 10,20
    loss = criterion(outputs, y_labeled) #outputs 10,2; y_labeled 10 (label from 0 to 1)
    loss.backward()
    optimizer.step()

    # Predict on the unlabeled data
    model.eval()
    with torch.no_grad():
        probs = model(X_unlabeled)  # 197,20, probs 197,2 (sum=1)
        uncertainties = 1 - torch.max(probs, dim=1).values  # Uncertainty = 1 - max probability

    # Query the most uncertain sample, query_sample 1,20; query_label 1
    query_idx = torch.argmax(uncertainties).item()  # uncertainties 197, prob
    query_sample, query_label = (X_unlabeled[query_idx].unsqueeze(0),
                                 y_unlabeled[query_idx].unsqueeze(0))

    # Add the queried sample to the labeled dataset
    X_labeled = torch.cat([X_labeled, query_sample])
    y_labeled = torch.cat([y_labeled, query_label])

    # Remove the queried sample from the unlabeled pool
    X_unlabeled = torch.cat([X_unlabeled[:query_idx], X_unlabeled[query_idx + 1:]])
    y_unlabeled = torch.cat([y_unlabeled[:query_idx], y_unlabeled[query_idx + 1:]])

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test.numpy(), y_pred.numpy())
        print(f"Iteration {i + 1}, Test Accuracy: {acc:.4f}")

print("Active learning complete.")
