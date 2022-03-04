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

warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)
sys.path.append("..")


from src.utils import PoseUtils
from src.dataset import YogaPosDataset
from src.train import TrainModel
from Models import base_gnn_model

class YogaPoseTrain(object):
    def __init__(self, base_path = None, device = None):
        self.base_path = BASEDIR if base_path is None else base_path
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.root_data_path = os.path.join(self.base_path, "Data/")
    
    def load_pretrained_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        return model 
    
    def load_data_loader(self, csv_name, batch_size = 64, shuffle = True):
        dataset = YogaPosDataset(self.root_data_path, csv_name)
        data_loader = DataLoader(
            dataset, 
            batch_size = 64,
            shuffle = True
        )

        return data_loader 
    
    def train(self, model, criterion, optimizer, epochs, train_loader, test_loader, valid_loader = None, model_save_name = None):
        self.train_model = TrainModel(model)

        for epoch in tqdm(range(1, epochs + 1)):
            current_train_loss, current_train_accuracy = self.train_model.train_model_perbatch(model, 
                                                                                               train_loader, 
                                                                                               criterion, optimizer)
            print(f"Epoch {epoch} Train Loss: {current_train_loss:.4f} Train Accuracy: {current_train_accuracy:.4f}")

            if valid_loader:
                current_valid_loss, current_valid_accuracy = self.train_model.evaluate_model(model, valid_loader)
                print(f"Epoch {epoch} Valid Loss: {current_valid_loss:.4f} Valid Accuracy: {current_valid_accuracy:.4f}")
        
        path_to_model_save = os.path.join(self.base_path, f"saved_models/{model_save_name}")
        torch.save(model.state_dict(), path_to_model_save)

        test_loss, test_acc = self.train_model.test_model_perbatch(model, test_loader, criterion)
        print(f"Test Loss: {test_loss} Test Accuracy: {test_acc}")


if __name__ == '__main__':
    yoga_pos = YogaPoseTrain()
    train_loader = yoga_pos.load_data_loader("train_data.csv")
    test_loader = yoga_pos.load_data_loader("test_data.csv")

    model = base_gnn_model.Model(3, 64, 16, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    yoga_pos.train(model, criterion, optimizer, 30, train_loader, test_loader)
