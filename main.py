import os 
import cv2 
import sys 
import warnings
import numpy as np 
import pandas as pd 
import mediapipe as mp 
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

warnings.filterwarnings("ignore")

path = os.getcwd()
sys.path.append(path)
sys.path.append(path[:-1])

from src.utils import PoseUtils
from src.dataset import YogaPosDataset
from src.train import TrainModel
from Models import base_gnn_model

device = "cuda" if torch.cuda.is_available() else "cpu"
root_data_path = os.path.join(path, "Data/")

train_dataset = YogaPosDataset(root_data_path, "train_data.csv")
test_dataset = YogaPosDataset(root_data_path, "test_data.csv", test = True, valid = False)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1,
    shuffle = True
)


model = base_gnn_model.Model(3, 64, 32, 5).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_model = TrainModel(model)

epochs = 30
for epoch in tqdm(range(epochs)):
    current_train_loss, current_train_acc = train_model.train_model_perbatch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch} Train Loss: {current_train_loss} Train Accuracy: {current_train_acc}")

model_save_path = os.path.join(os.getcwd(), "saved_models/base_model.pth")
torch.save(model.state_dict(), model_save_path)

test_loss, test_acc = train_model.test_model_perbatch(model, test_loader, criterion)
print(f"Test Loss: {test_loss} Test Accuracy: {test_acc}")