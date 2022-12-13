import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(BASEDIR))
sys.path.append("..")

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrainModel(nn.Module):
    def __init__(self, model):
        pass

    def train_model_perbatch(
        self, model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, optimizer
    ):
        """
        Function to train the model for a single epoch or train all the batches
        args:
        ----
        model : (torch.nn.Module) Pytorch model
        loader : (torch.utils.data.DataLoader) The DataLoader to train from
        criterion : (torch.nn.Module) The loss Function
        optimizer : (torch.optim) The optimizer

        returns:
        -------
        This will return the mean batch loss (or a loss for 1 single epoch) and the accuracy of the model
        """
        running_loss = 0.0
        num_correct = 0.0

        model.train()
        for graph in loader:
            x = graph.x.to(device)
            y = graph.y.type(torch.long).to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)

            optimizer.zero_grad()
            output = model(x, edge_index, batch)
            num_correct += int((output.argmax(dim=1) == y).sum())

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(loader.dataset), num_correct / len(loader.dataset)  # type: ignore

    def test_model_perbatch(
        self, model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module
    ):
        """
        Function to test the model for a single epoch
        args:
        ----
        model : (torch.nn.Module) Pytorch model
        loader : (torch.utils.data.DataLoader) The DataLoader to train from
        criterion : (torch.nn.Module) The loss Function

        returns:
        -------
        This will return the mean batch loss (or a loss for 1 single epoch) and the accuracy of the model
        """
        model.eval()
        running_loss = 0.0
        num_correct = 0.0

        with torch.no_grad():
            for graph in loader:
                x = graph.x.to(device)
                y = graph.y.type(torch.long).to(device)
                edge_index = graph.edge_index.to(device)
                batch = graph.batch.to(device)

                output = model(x, edge_index, batch)
                loss = criterion(output, y)
                running_loss += loss.item()
                num_correct += int((output.argmax(dim=1) == y).sum())
        return running_loss / len(loader.dataset), num_correct / len(loader.dataset)  # type: ignore

    def evaluate_model(self, model: torch.nn.Module, loader: DataLoader):
        """
        This will return the accuracy of the model for a very single batch
        args:
        ----
        model : (torch.nn.Module) Pytorch model
        loader : (torch.utils.data.DataLoader) The DataLoader to train from
        """
        with torch.no_grad():
            graph = next(iter(loader))
            x = graph.x.to(device)
            y = graph.y.type(torch.long).to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)

            output = model(x, edge_index, batch)
            return int((output.argmax(dim=1) == y).sum())
