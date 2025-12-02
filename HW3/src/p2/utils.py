import torch

def save_adapters(model: torch.nn.Module, path: str):
    """
    Save only LoRA and visual_projection params to keep checkpoint small.
    """
    state = model.state_dict()
    keep = {}
    for k, v in state.items():
        if "lora" in k or "visual_projection" in k or "lora_up" in k or "lora_down" in k:
            keep[k] = v.cpu()
    torch.save(keep, path)

def load_adapters(model: torch.nn.Module, path: str, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)

def count_trainable_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
