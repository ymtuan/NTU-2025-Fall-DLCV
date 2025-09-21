import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

import tsne_utils

model_path = "../pretrained/checkpoint0500.pth"

class OfficeHomeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx, 2])
        if self.transform:
            image = self.transform(image)
        return image, label

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples
    
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        def __call__(self, val_acc):
            if self.best_score is None:
                self.best_score = val_acc
            elif val_acc < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_acc
                self.counter = 0

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 finetune.py <train.csv> <train_folder> <val_csv> <val_folder>")
        return

    train_csv = sys.argv[1]
    train_dir = sys.argv[2]
    val_csv = sys.argv[3]
    val_dir = sys.argv[4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = OfficeHomeDataset(train_csv, train_dir, transform_train)
    val_dataset = OfficeHomeDataset(val_csv, val_dir, transform_val)

#   print(len(set(train_dataset.data['label'])))
#   print(len(set(val_dataset.data['label'])))
#   print(train_dataset.data['label'].value_counts())
#   print(val_dataset.data['label'].value_counts())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    backbone = models.resnet50()
    backbone.fc = nn.Identity()

    checkpoint = torch.load(model_path, map_location="cpu")
    
    if 'student' in checkpoint:
        student_state_dict = checkpoint['student']
        state_dict = {}
        for k, v in student_state_dict.items():
            if k.startswith('module.backbone.'):
                new_k = k[len('module.backbone.'):]
            else:
                new_k = k
            state_dict[new_k] = v
    else:
        state_dict = checkpoint

    msg = backbone.load_state_dict(state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    backbone = backbone.to(device)

    num_classes = 65
    classifier = nn.Linear(2048, num_classes).to(device)

    # Config 1: train backbone + classifier
    # Config 2: freeze backbone, train only classifier
    config = 1
    if config == 1:
        model = nn.Sequential(backbone, classifier).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    elif config == 2:
        for param in backbone.parameters():
            param.requires_grad = False
        model = nn.Sequential(backbone, classifier).to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = nn.CrossEntropyLoss()
    num_epochs = 25
    save_every = 25
    save_dir = "./finetune_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # early_stopping = EarlyStopping(patience=10)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"{save_dir}/finetune_model_epoch{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch}/{num_epochs}")

        if epoch == 0 or epoch == num_epochs - 1:
            features, labels = [], []
            backbone.eval()
            with torch.no_grad():
                for imgs, lbls in train_loader:
                    imgs = imgs.to(device)
                    feat = backbone(imgs)
                    features.append(feat.cpu())
                    labels.append(lbls)
            features = torch.cat(features)
            labels = torch.cat(labels)

            tsne_utils.tsne_visualization(model, train_loader, device, epoch)

if __name__ == "__main__":
    main()