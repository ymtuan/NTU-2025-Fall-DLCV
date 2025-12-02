import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Check about the root
DEFAULT_MODEL_PATH = "checkpoints/p1.pth"

class OfficeHomeTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 1]  # 'filename' column
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, filename

def inference(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, filenames in tqdm(dataloader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().tolist()
            for f, p in zip(filenames, preds):
                results.append((f, p))
    return results

def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python3 p1_inference.py <test.csv> <img_folder> <output_csv> [model_path]")
        return

    test_csv = sys.argv[1]
    img_dir = sys.argv[2]
    output_csv = sys.argv[3]
    model_path = sys.argv[4] if len(sys.argv) == 5 else DEFAULT_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    dataset = OfficeHomeTestDataset(test_csv, img_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    # Model
    backbone = models.resnet50()
    backbone.fc = nn.Identity()
    num_classes = 65
    classifier = nn.Linear(2048, num_classes)
    model = nn.Sequential(backbone, classifier).to(device)

    # Load checkpoint (both backbone + classifier)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Run inference
    predictions = inference(model, loader, device)

    # Save predictions
    df = pd.read_csv(test_csv)
    df['label'] = [p for _, p in predictions]
    df.to_csv(output_csv, index=False)
    print(f"Inference finished. Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
