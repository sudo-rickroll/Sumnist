import torch
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

def data_loader(dataset:torch.utils.data.Dataset, 
               batch_size : int = 1, 
               shuffle : bool =False, 
               num_workers : int = 0,
               pin_memory : bool =False
               ) -> torch.utils.data.DataLoader :

    return torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)