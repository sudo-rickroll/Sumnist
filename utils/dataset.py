import torch
import torchvision
import random
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

class MNISTSumDataset(torch.utils.data.Dataset):
    def __init__(self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False):
      
        self.data = torchvision.datasets.MNIST(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.rand = [random.randint(0,9) for _ in range(len(self.data))]
        
    def __getitem__(self, index):
        img, label, rand, target = self.data[index][0], int(self.data[index][1]), self.rand[index], int(self.data[index][1]) + self.rand[index]
        
        return img, label, rand, target
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
      return self.data.__repr__()