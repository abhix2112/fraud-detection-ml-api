import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# --------------------------
# 1. Load Dataset
# --------------------------

df = pd.read_csv("creditcard.csv")

print("Fraud %:", df["Class"].mean() * 100)

X = df.drop("Class", axis=1)
y = df["Class"]

# --------------------------
# 2. Scale Data
# --------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train.values).unsqueeze(1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test.values).unsqueeze(1)

# --------------------------
# 3. Neural Network Model
# --------------------------

class FraudNet(nn.Module):

    def __init__(self, input_size):
        super(FraudNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


model = FraudNet(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 4. Training
# --------------------------

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

epochs = 20

for epoch in range(epochs):

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# --------------------------
# 5. Evaluation
# --------------------------

with torch.no_grad():

    predictions = model(X_test)

    predicted_labels = (predictions > 0.5).numpy()

    probs = predictions.numpy()

    y_true = y_test.numpy()

print("\nConfusion Matrix")
print(confusion_matrix(y_true, predicted_labels))

print("\nClassification Report")
print(classification_report(y_true, predicted_labels))

# ROC-AUC
auc = roc_auc_score(y_true, probs)

print("\nROC-AUC:", auc)

# --------------------------
# 6. Save Model + Scaler
# --------------------------

torch.save(model.state_dict(), "fraud_model.pth")

joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")