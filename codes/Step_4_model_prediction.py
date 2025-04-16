##Creator: Xinghua cheng
##Date: 16/04/2025
##Contact: xinghua.cheng@uconn.edu

##Goal: to train a neural network for predicting GPP
##Note: xinghua used ChatGPT 4.0 to enhance programming.

import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## get data from .npz format
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


def get_test_loader(npz_file, batch_size=32):
    test_ds = GPPDatasetFromNPZ(npz_file)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def evaluate_model(model, loader, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)

    preds, targets = [], []

    with torch.no_grad():
        for images, aux_data, target in loader:
            images, aux_data = images.to(device), aux_data.to(device)
            outputs = model(images, aux_data).cpu().numpy().flatten()
            preds.extend(outputs)
            targets.extend(target.numpy().flatten())

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"\n📊 Test Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    return rmse, mae, r2

## the ResNet Model
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

## the main function of model test
def test_main_function(testNPZ,checkpoint_path):
        
    test_loader = get_test_loader(testNPZ) ## for the year of 2020
    model = ResNetWithenvData(in_channels=6, env_dim=10) ## it should be the same as the trained neural network

    evaluate_model(model, test_loader, checkpoint_path)

    return 1

if __name__ == "__main__":
    print("Evaluating on test set...")
    testNPZ = "/home/xic23015/gpp_exercise/data/gpp_dataset_test.npz" #it is obtained from "Step_4_Prepare_Test_Data.py"
    checkpoint_path = "/home/xic23015/gpp_exercise/checkpoints/model_random_split_epoch40.pth" # model_leave_out_2021_epoch20.pth
    ##
    test_main_function(testNPZ,checkpoint_path)
