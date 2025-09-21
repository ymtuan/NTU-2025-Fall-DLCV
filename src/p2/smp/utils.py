import torch
import torch.nn as nn

def get_loss():
    """
    CrossEntropyLoss, ignoring 'Unknown' (class 6).
    """
    return nn.CrossEntropyLoss(ignore_index=6)

def save_model(model, path="best_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="best_model.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
