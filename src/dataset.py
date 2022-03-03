import os
import sys
import warnings
import numpy as np
import pandas as pd  
from tqdm import tqdm
import mediapipe as mp 
from pathlib import Path

import torch
import torch_geometric
from torch_geometric.data import Dataset 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)
sys.path.append("..")

class YogaPosDataset(Dataset):
    def __init__(self, root, file_name, valid = False, test = False, transform = None, pre_transform = None, add_self_loops = False, normalize_edge = False):
        self.test = test 
        self.valid = valid
        self.file_name = file_name
        
        pose_indices = [
                (0, 1),
                (0, 6),
                (0, 2),
                
                (1, 3),
                (1, 7),
                (1, 0),
                
                (2, 4),
                (2, 0),
                
                (3, 5),
                (3, 1),
                
                (4, 2),
                (5, 3),
                
                (6, 0),
                (6, 7),
                (6, 8),
                
                (7, 1),
                (7, 6),
                (7, 9),
                
                (8, 6),
                (8, 10),
                
                (9, 7),
                (9, 11),
                
                (10, 8),
                (10, 12),
                
                (11, 9),
                (11, 13),
                
                (12, 10),
                (13, 11)
            ]


        source_indices = []
        target_indices = []

        for pos in pose_indices:
            source_indices.append(pos[0])
            target_indices.append(pos[1])

        edge_indices = np.array([
            source_indices,
            target_indices
        ], dtype=np.float32)



        edge_index = torch.tensor(edge_indices, dtype=torch.long)

        if add_self_loops:
            edge_index = torch_geometric.utils.add_self_loops(edge_index)
        
        if normalize_edge:
            edge_index = edge_index / 14

        self.edge_index = edge_index

        super(YogaPosDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return self.file_name
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index(drop=True)
        if "Unnamed: 0" in self.data.columns:
            self.data = self.data.drop(['Unnamed: 0'], axis=1)
        
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        
        elif self.valid:
            return [f'data_valid_{i}.pt' for i in list(self.data.index)]
        
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def download(self):
        pass 

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.long)
    
    def len(self):
        return self.data.shape[0]
    
    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        
        elif self.valid:
            data = torch.load(os.path.join(self.processed_dir, f'data_valid_{idx}.pt'))

        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

        return data 
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)
        if "Unnamed: 0" in self.data.columns:
            self.data = self.data.drop(['Unnamed: 0'], axis=1)
        
        for index, pos in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            features = np.array(pos.iloc[:-2], dtype=np.float32).reshape(14, 3)
            x = torch.tensor(features, dtype=torch.float32)
            y = self._get_label(pos.iloc[-1]) 
            data = torch_geometric.data.Data(
                x = x,
                edge_index = self.edge_index,
                y = y
            )

            if self.test:
                torch.save(
                    data,
                    os.path.join(self.processed_dir, f'data_test_{index}.pt')
                )
            elif self.valid:
                torch.save(
                    data,
                    os.path.join(self.processed_dir, f'data_valid_{index}.pt')
                )

            else:
                torch.save(
                    data,
                    os.path.join(self.processed_dir, f'data_{index}.pt')
                )


if __name__ == '__main__':
    path = BASEDIR
    root_data_path = os.path.join(path, "Data/")
    train_dataset = YogaPosDataset(root_data_path, "train_data.csv")
    test_dataset = YogaPosDataset(root_data_path, "test_data.csv", test = True, valid = False)

    print(train_dataset)
    print(test_dataset)
    print()

    print(train_dataset[0])
    print(test_dataset[0])
    print()