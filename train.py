import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pickle

from model import Seq2Seq
from dataset import ChatDataset
from preprocess import tokenize

# === Hyperparameters ===
BATCH_SIZE = 16
EPOCHS = 50
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
MAX_LEN = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
PAD_IDX = 0

# === Load Data ===
with open("model/pairs.pkl", "rb") as f:
    pairs = pickle.load(f)

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word2index = tokenizer.word2index
index2word = tokenizer.index2word
vocab_size = len(word2index)

# === Filter out long pairs ===
filtered_pairs = [
    (x, y) for x, y in pairs
    if len(tokenize(x)) <= MAX_LEN and len(tokenize(y)) <= MAX_LEN
]

print(f"✅ Total usable training pairs: {len(filtered_pairs)}")

# === Split into Train/Val ===
val_ratio = 0.1
val_size = int(len(filtered_pairs) * val_ratio)
train_size = len(filtered_pairs) - val_size
train_data, val_data = random_split(filtered_pairs, [train_size, val_size])

train_loader = DataLoader(ChatDataset(train_data, word2index), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChatDataset(val_data, word2index), batch_size=BATCH_SIZE)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2Seq(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Early Stopping Setup ===
best_val_loss = float('inf')
no_improve_epochs = 0

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, targets, teacher_forcing_ratio=0.5, max_len=MAX_LEN)

        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, targets, teacher_forcing_ratio=0.0, max_len=MAX_LEN)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"📊 Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # === Save Best Model ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), "model/chatbot_model.pt")
        print("✅ Best model updated and saved.")
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement. Patience: {no_improve_epochs}/{EARLY_STOPPING_PATIENCE}")

    # === Early Stopping ===
    if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print("🛑 Early stopping triggered.")
        break

# === Save Full Model (optional) ===
torch.save(model, "model/full_chatbot_model.pth")
print("✅ Full model saved at model/full_chatbot_model.pth")
