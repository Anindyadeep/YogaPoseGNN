import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader  # type: ignore
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(BASEDIR))
sys.path.append("..")

from Models import base_gnn_model

from src.dataset import YogaPosDataset
from src.train import TrainModel
from src.utils import PoseUtils


class YogaPoseTrain(object):
    def __init__(self, base_path: Optional[str] = None, device: Optional[str] = None):
        """
        args:
        ----
        base_path : (str) the root path of the project
        device : (str) the device to train the model
        """
        self.base_path = BASEDIR if base_path is None else base_path
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.root_data_path = os.path.join(self.base_path, "Data/")

    def load_pretrained_model(self, model: torch.nn.Module, model_path: str):
        """
        Loads a pretrained PyTorch model which has an extension of .pth
        args:
        ----
        model : (torch.nn.Module) The PyTorch model
        model_path : (str) The path of the pretrained weights and the configuration of the model

        returns:
        --------
        PyTorch (torch.nn.Module) model object with pretrained weights
        """
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        return model

    def load_data_loader(
        self, csv_name: str, batch_size: Optional[int] = 64, shuffle: Optional[bool] = True
    ):
        """
        Loads the PyTorch DataLoader
        args:
        ----
        csv_name : (str) The name of the CSV file name to create the dataloader
        batch_size : (int) The batch size of the dataloader
        shuffle : (boolean) Whether to shuffle the datapoints or not

        returns:
        --------
        Returns the PyTorch (torch.utils.data.DataLoader) object
        """
        dataset = YogaPosDataset(self.root_data_path, csv_name)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader

    def train(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        model_save_name: Optional[str] = None,
    ):
        """
        Custom Training function to train the existing model or a new model
        args:
        -----
        model : (torch.nn.Module) The PyTorch model
        criterion : (torch.nn.Module) The loss Function
        optimizer : (torch.optim) The optimizer
        epochs : (int) The number of iteration to train the model
        train_loader : (torch.utils.data.DataLoader) The Train DataLoader
        test_loader : (torch.utils.data.DataLoader) The Test DataLoader
        valid_loader : (torch.utils.data.DataLoader) The Validation DataLoader (not required if not mandatory)
        model_save_name : (str) The name of the model to save in .pth path
        """
        self.train_model = TrainModel(model)

        for epoch in tqdm(range(1, epochs + 1)):
            current_train_loss, current_train_accuracy = self.train_model.train_model_perbatch(
                model, train_loader, criterion, optimizer
            )
            print(
                f"Epoch {epoch} Train Loss: {current_train_loss:.4f} Train Accuracy: {current_train_accuracy:.4f}"
            )

            if valid_loader:
                current_valid_loss, current_valid_accuracy = self.train_model.evaluate_model(model, valid_loader)  # type: ignore
                print(
                    f"Epoch {epoch} Valid Loss: {current_valid_loss:.4f} Valid Accuracy: {current_valid_accuracy:.4f}"
                )

        path_to_model_save = os.path.join(self.base_path, f"saved_models/{model_save_name}")
        torch.save(model.state_dict(), path_to_model_save)

        test_loss, test_acc = self.train_model.test_model_perbatch(model, test_loader, criterion)
        print(f"Test Loss: {test_loss} Test Accuracy: {test_acc}")
