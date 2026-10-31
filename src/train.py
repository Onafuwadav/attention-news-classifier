import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

from dataset import CustomTextDataset
from model import NeuralNetwork
from glove import load_glove_matrix

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Use GPU if available or the MacOS MPS equivalent
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using {} device".format(device))

# Switch dataset and embeddings
parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=["full", "sample"], default="full",
                    help="run download_data.py")
parser.add_argument("--no-glove", action="store_true", help="train word embeddings from scratch")
parser.add_argument("--epochs", type=int, default=None)
args = parser.parse_args()

torch.manual_seed(1)


if args.data == "full":
    all_data_object = CustomTextDataset(ROOT / "data" / "News_Category_Dataset_v3.json", verbose=True,
                                        use_description=True, max_len=40, min_freq=3, merge_categories=True)
else:
    all_data_object = CustomTextDataset(ROOT / "data" / "NewsHeadlines.csv", verbose=True, use_description=True)

# Training config
learning_rate = 0.001
batch_size = 256 if args.data == "full" else 32
epochs = args.epochs or (5 if args.data == "full" else 60)
train_prop = 0.8      # standard 80 / 20 split
weight_decay = 1e-4 if args.data == "sample" else 0

train_no = round(len(all_data_object) * train_prop)
val_no = len(all_data_object) - train_no

maxSentenceLength = all_data_object.max_sentence_len + 1

vocabSize = len(all_data_object.unique_tokens)
nCategories = len(set(all_data_object.category))

print(f'no train: {train_no}, no val: {val_no}')
train_subset, val_subset = torch.utils.data.random_split(all_data_object, [train_no, val_no], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size)

# Model
wordEmbedDim = 100   # to  match GloVe
embedDim = 64 if args.data == "full" else 32
noHeads = 4 if args.data == "full" else 2
hiddenDim = 128 if args.data == "full" else 64
dropout = 0.3

# Initialise embedding layer of pretrained GloVe vectors
pretrained = None
if not args.no_glove:
    pretrained, coverage = load_glove_matrix(ROOT / "data" / "glove.6B.100d.txt",
                                             all_data_object.unique_tokens, all_data_object.pad_idx)
    print(f"GloVe covers {100 * coverage:.1f}% of the {vocabSize} word vocabulary")

model = NeuralNetwork(maxSentenceLength, vocabSize, wordEmbedDim,
    embedDim, noHeads, hiddenDim, nCategories, dropout, all_data_object.pad_idx, pretrained)
model = model.to(device)

print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

## Loss and optimiser(rare categories count as more)
train_labels = all_data_object.category_int[train_subset.indices].numpy()
counts = np.bincount(train_labels, minlength=nCategories)
class_weights = len(train_labels) / (nCategories * np.maximum(counts, 1))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)


optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_loop(dataloader, model, loss_fn, optimiser):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # NEW: move batch to the same device as the model
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        total_loss += loss.item()
    # Return loss for plotting
    return total_loss / len(dataloader)


# Loop epoch for performance test on validation data
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.append(pred.argmax(1).cpu())
            all_targets.append(y.cpu())

    test_loss /= num_batches
    correct /= size
    return test_loss, correct, torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


# Track and keep best model
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_f1, best_state, best_epoch = -1, None, 0

for t in range(epochs):
    train_loss = train_loop(train_loader, model, loss_fn, optimiser)
    val_loss, val_acc, preds, targets = test_loop(val_loader, model, loss_fn)
    val_f1 = f1_score(targets, preds, average="macro", zero_division=0)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    if val_f1 > best_f1:
        best_f1, best_epoch = val_f1, t + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epochs <= 20 or (t + 1) % 5 == 0 or t == 0:
        print(f"Epoch {t+1:>3} | train loss {train_loss:.3f} | val loss {val_loss:.3f} "
              f"| val acc {100*val_acc:5.1f}% | val macro-F1 {val_f1:.3f}")

print("Done!")

model.load_state_dict(best_state)
val_loss, val_acc, preds, targets = test_loop(val_loader, model, loss_fn)

majority = Counter(train_labels).most_common(1)[0][0]
baseline_acc = (targets == majority).mean()

print(f"\nBest epoch: {best_epoch}")
print(f"Validation accuracy: {100*val_acc:.1f}%  (majority-class baseline: {100*baseline_acc:.1f}%)")
print(f"Validation macro-F1: {f1_score(targets, preds, average='macro', zero_division=0):.3f}")
present = sorted(set(targets) | set(preds))
print(classification_report(targets, preds, labels=present,
                            target_names=[all_data_object.class_names[i] for i in present],
                            zero_division=0))

OUTPUT_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="validation")
axes[0].set_title("Weighted cross-entropy loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history["val_acc"], label="accuracy")
axes[1].plot(history["val_f1"], label="macro-F1")
axes[1].set_title("Validation metrics")
axes[1].set_xlabel("Epoch")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
plot_file = OUTPUT_DIR / f"training_curves_{args.data}{'_noglove' if args.no_glove else ''}.png"
fig.savefig(plot_file, dpi=120)
print(f"Saved {plot_file}")