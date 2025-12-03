import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import datetime

from ranger21 import Ranger21
from Dataset import NewsDataset, load_data
from nnModel import BiLSTM_Attention_Model


embeddings, classes = load_data()

X = torch.tensor(embeddings, dtype=torch.float32)
y = torch.tensor(classes, dtype=torch.long).squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

input_dim = X.shape[1]
num_classes = 2

model = BiLSTM_Attention_Model(
    input_dim=input_dim,
    hidden_dim=256,
    num_classes=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
num_epochs = 150


optimizer = Ranger21(
    model.parameters(),
    lr=3e-4,             
    weight_decay=1e-3,
    num_epochs=num_epochs,
    num_batches_per_epoch=len(train_loader)
)

global_loss = float("inf")

version = "v1"
model_path = f"./models/yaari-synth-auditor-model-{version}.pth"

with open(f"./models/logs-{version}.txt", "a") as file:

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_X)

            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        log = (
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} [{datetime.datetime.now()}]"
        )

        if avg_loss < global_loss:
            global_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            log += f"\n{model_path} updated with new weights."

        file.write(log + "\n")
        print(log)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs, _ = model(batch_X)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))
