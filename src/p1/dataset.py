import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

class DigitStyleDataset(Dataset):
    def __init__(self, mnistm_dir='../../hw2_data/digits/mnistm', svhn_dir='../../hw2_data/digits/svhn', transform=None):
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load MNIST-M (even labels)
        mnistm_csv = pd.read_csv(os.path.join(mnistm_dir, 'train.csv'))
        self.mnistm_data = [(os.path.join(mnistm_dir, 'data', row['image_name']), row['label']) 
                           for _, row in mnistm_csv.iterrows() if row['label'] % 2 == 0]
        
        # Load SVHN (odd labels)
        svhn_csv = pd.read_csv(os.path.join(svhn_dir, 'train.csv'))
        self.svhn_data = [(os.path.join(svhn_dir, 'data', row['image_name']), row['label']) 
                         for _, row in svhn_csv.iterrows() if row['label'] % 2 == 1]
        
        self.data = self.mnistm_data + self.svhn_data
        print(f"Loaded {len(self.mnistm_data)} MNIST-M (even) and {len(self.svhn_data)} SVHN (odd) images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)