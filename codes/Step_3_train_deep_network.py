##Creator: Xinghua Cheng
##Contact: xinghua.cheng@uconn.edu
##Date: April 16, 2025

##Goal: to train a neural network for predicting GPP
##Note: xinghua used ChatGPT 4.0 to enhance programming.

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18, ResNet18_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## save the trained model for each epoch
def save_checkpoint(model, fold_name, epoch):
    os.makedirs("checkpoints", exist_ok=True)
    path = f"/home/xic23015/gpp_exercise/checkpoints/model_{fold_name}_epoch{epoch}.pth"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")

## plot the loss curve
def plot_loss(train_losses, val_losses, fold_name):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Curve ({fold_name})")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"/home/xic23015/gpp_exercise/plots/loss_curve_{fold_name}.png", dpi=300)
    plt.close()

def split_by_year(data_dict, test_year):
    train_mask = data_dict['year'] != test_year
    test_mask = data_dict['year'] == test_year

    X_train = data_dict['X_img'][train_mask]
    X_env_train = data_dict['X_env'][train_mask]
    y_train = data_dict['y'][train_mask]

    X_val = data_dict['X_img'][test_mask]
    X_env_val = data_dict['X_env'][test_mask]
    y_val = data_dict['y'][test_mask]

    return (X_train, X_env_train, y_train), (X_val, X_env_val, y_val)

class ResNetWithenvData(nn.Module):
    def __init__(self, in_channels=10, env_dim=5):  # 5 = number of MERRA-2 features
        super(ResNetWithenvData, self).__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn_out_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # we'll handle the head manually

        self.mlp = nn.Sequential(
            nn.Linear(env_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, image, env_data):
        cnn_feat = self.cnn(image)           # from HLS
        env_feat = self.mlp(env_data)        # from MERRA-2
        combined = torch.cat([cnn_feat, env_feat], dim=1)
        return self.head(combined)

## the class for reading data in ".npz" format
class GPPDatasetFromNPZ(Dataset):
    def __init__(self, npz_file, indices=None):
        data = np.load(npz_file)
        self.X_img = data['X_img']
        self.X_aux = data['X_aux']
        self.y = data['y']
        self.year = data['year']

        if indices is not None:
            self.X_img = self.X_img[indices]
            self.X_aux = self.X_aux[indices]
            self.y = self.y[indices]
            self.year = self.year[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = torch.tensor(self.X_img[idx], dtype=torch.float32)   # shape (6, 50, 50)
        aux = torch.tensor(self.X_aux[idx], dtype=torch.float32)   # shape (10,)
        target = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)  # shape (1,)
        return img, aux, target

##train the model
def train_model(model, train_loader, val_loader, fold_name, epochs=20):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for images, env_data, targets in train_loader:
            images, env_data, targets = images.to(device), env_data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, env_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, env_data, targets in val_loader:
                images, env_data, targets = images.to(device), env_data.to(device), targets.to(device)
                outputs = model(images, env_data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"[{fold_name}] Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

        # Save checkpoint
        save_checkpoint(model, fold_name, epoch+1)

    # Plot training curve
    plot_loss(train_losses, val_losses, fold_name)

## The function to read npz file and obtain hls and merra data according to the test year
def get_random_split_data_loaders(npz_file, included_years, val_ratio=0.2, batch_size=32, seed=42):
    data = np.load(npz_file)
    all_years = data['year']

    # Keep only samples from included years
    mask = np.isin(all_years, included_years)
    indices = np.where(mask)[0]

    # Split into train and val
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed)

    train_ds = GPPDatasetFromNPZ(npz_file, indices=train_idx)
    val_ds   = GPPDatasetFromNPZ(npz_file, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    print(f"[Random Split] Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader

## the main function for traning a model
def train_model_main(yearLis):
    # Full data dictionary (replace with your real data)
    data_dict = {
        'X_img': np.array([...]),        # shape: (N, 6, 50, 50)
        'X_env': np.array([...]),        # shape: (N, 10)
        'y': np.array([...]),            # shape: (N,)
        'year': np.array([...]),         # shape: (N,) with values in [2018–2021]
    }

    train_loader, val_loader = get_random_split_data_loaders(
        "/home/xic23015/gpp_exercise/data/gpp_dataset_train.npz",
        included_years=yearLis,
        val_ratio=0.2,
        batch_size=32
    )

    model = ResNetWithenvData(in_channels=6, env_dim=10)
    train_model(model, train_loader, val_loader, fold_name="random_split", epochs=40)

    return 1

if __name__ == "__main__":

    # Dummy data for illustration — replace with real data
    print("We are traninig a deep neural network for predicting GPP")

    yearLis = [2018, 2019, 2021]

    train_model_main(yearLis)

    print("The deep neural network has been trained")
