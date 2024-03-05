import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class RAF_DB_Loader():
    def __init__(self, data_dir, batch_size = 5, resize = 224):
        self.dataset = datasets.ImageFolder(data_dir, transforms.Compose([transforms.Resize((resize, resize)), # 调整大小
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]), 
                    # transforms.RandomErasing(scale=(0.02, 0.1)),])) 
                    transforms.RandomErasing(scale=(0.02, 0.25)),]))                                                       
        self.loader = torch.utils.data.DataLoader(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           pin_memory=True)
        
    def get_loader(self):
        return self.loader