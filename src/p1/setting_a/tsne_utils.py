import torch
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def get_feature_extractor(model):
    """Return backbone without final classifier layer."""
    return nn.Sequential(*list(model.children())[:-1])

def extract_features(model, dataloader, device):
    """Extarct second-last layer features + labels."""
    model.eval()
    extractor = get_feature_extractor(model).to(device)

    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feat = extractor(x)
            feat = torch.flatten(feat, 1)
            features.append(feat.cpu())
            labels.append(y)
    return torch.cat(features), torch.cat(labels)

def tsne_visualization(model, dataloader, device, epoch, save_dir="tsne_plots"):
    """
    Run t-SNE on model features, plot and save to file.
    Filename format: epoch_{epoch}_tsne.png
    """
    features, labels = extract_features(model, dataloader, device)
    features = features.numpy()
    labels = labels.numpy()

    tsne = TSNE(n_components=2, init='pca', random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=10)
    plt.legend(*scatter.legend_elements(), title="classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"t-SNE at Epoch {epoch}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_tsne.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to {save_path}")