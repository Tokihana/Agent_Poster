import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torchsampler import ImbalancedDatasetSampler

class Val_Loader():
    def __init__(self, data_dir, batch_size = 5, resize = 224, workers=4):
        self.dataset = datasets.ImageFolder(data_dir,
                                            transforms.Compose([transforms.Resize((224, 224)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225]),]))
        self.loader = torch.utils.data.DataLoader(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=workers,
                                           pin_memory=True) # depends on device, set True if have enough memories
        
    def get_loader(self):
        return self.loader