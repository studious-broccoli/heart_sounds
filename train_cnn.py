import pdb
from data.config import *
import torch
from torch.utils.data import DataLoader
from data.dataset import HeartSoundDataset
from models.cnn import HeartSoundCNN
from utils.plot_utils import visualize_feature_maps, visualize_saliency, plot_losses, basic_cm_snsp
from data.load_data import load_labels
import glob
import torch.nn as nn
from tqdm import tqdm

import multiprocessing as mp
mp.set_start_method('fork', force=True)


def run_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)):
        optimizer.zero_grad()
        mfccs, labels = batch["mfccs"], batch["labels"]
        mfccs, labels = mfccs.to(device), labels.to(device)

        outputs = model(mfccs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # if epoch == 9 and batch_idx == 0:
        #     validate_batch(model, mfccs, device, epoch)

    return running_loss / len(loader)


def validate_batch(model, mfccs, device, epoch):
    model.eval()  # turn off dropout, batchnorm in eval mode
    with torch.no_grad():
        mfcc_tensor = mfccs[0].unsqueeze(0).to(device)  # [1, 1, 13, 300]
        pred_label = model(mfcc_tensor).argmax(dim=1).item()

    visualize_feature_maps(model, mfcc_tensor, filename=f"./results/feature_map_{epoch}.png")
    visualize_saliency(model, mfcc_tensor.clone().requires_grad_(), label_idx=pred_label,
                       filename=f"./results/feature_saliency_{epoch}.png")


def validate_epoch(model, val_loader, device):
    correct = 0
    total = 0
    for batch in val_loader:
        mfccs, labels = batch["mfccs"].to(device), batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(mfccs)
            _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")


def evaluate_final(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    predicted_all = []
    labels_all = []

    for batch in val_loader:
        mfccs = batch["mfccs"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(mfccs)
            _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        predicted_all.extend(predicted.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    val_acc = 100 * correct / total
    print(f"Final Accuracy: {val_acc:.2f}%")

    # Convert to label strings
    y_test = [int_to_label_01[l] for l in labels_all]
    y_pred = [int_to_label_01[p] for p in predicted_all]

    basic_cm_snsp(y_test, y_pred, filename=f"./results/confusion_matrix_{DATA_TYPE}_{MODEL_TYPE}.png")


def train_cnn():
    # Load your data
    y_train = load_labels(TRAIN_LABELS_PATH, count=N_train)
    y_val = load_labels(TEST_LABELS_PATH, count=N_test)
    # y_train = [int_to_label[label] for label in y_train]
    # y_val = [int_to_label[label] for label in y_val]

    X_train = sorted(glob.glob(TRAIN_AUDIO_PATH))[:N_train]
    X_val = sorted(glob.glob(TEST_AUDIO_PATH))[:N_test]

    # Set up datasets and dataloaders
    train_dataset = HeartSoundDataset(X_train, y_train)
    val_dataset = HeartSoundDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeartSoundCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epoch_losses = []
    for epoch in range(NUM_EPOCHS):

        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, epoch)

        epoch_losses.append(train_loss)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}')

        if epoch % 5 == 0:
            validate_epoch(model, val_loader, device)

    plot_losses(epoch_losses, filename="results/training_loss.png")
    evaluate_final(model, val_loader, device)


if __name__ == '__main__':
    train_cnn()
