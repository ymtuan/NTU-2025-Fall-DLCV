import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Dataset
# -----------------------------
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
        if self.transform:
            image = self.transform(image)
        return image, filename

# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python3 p1_inference.py <test.csv> <img_folder> <output_csv> [model_path]")
        return

    test_csv = sys.argv[1]
    img_dir = sys.argv[2]
    output_csv = sys.argv[3]

    # Use model path from argument if provided; else use hardcoded default
    if len(sys.argv) == 5:
        model_path = sys.argv[4]
    else:
        model_path = "./pretrained/checkpoint0500.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_dataset = OfficeHomeDataset(test_csv, img_dir, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # -----------------------------
    # Load backbone
    # -----------------------------
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

    backbone.load_state_dict(state_dict, strict=False)
    backbone = backbone.to(device)
    backbone.eval()

    # -----------------------------
    # Classifier
    # -----------------------------
    num_classes = 65
    classifier = nn.Linear(2048, num_classes).to(device)
    model = nn.Sequential(backbone, classifier).to(device)
    model.eval()

    # -----------------------------
    # Inference
    # -----------------------------
    results = []
    with torch.no_grad():
        for imgs, filenames in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            for fname, pred in zip(filenames, preds):
                results.append([fname, pred])

    # -----------------------------
    # Save predictions
    # -----------------------------
    df_out = pd.DataFrame(results, columns=['file', 'label'])
    df_out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

if __name__ == "__main__":
    main()
