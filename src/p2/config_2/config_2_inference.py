import json
import torch
from torch.utils.data import DataLoader
from dataset import ImageCaptionDataset, collate_fn
from vit_encoder import ViTEncoder
from decoder_lora import Decoder, Config
from utils import load_adapters
from tokenization_qwen3 import Qwen3Tokenizer
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
images_root = "../../hw3_data/p2_data/images/val/"
ann_file = "../../hw3_data/p2_data/val.json"
vocab_file = "vocab.json"
merges_file = "merges.txt"
adapter_path = "checkpoints/adapters_epoch4.pt"
r = 16
alpha = 64

def main():
    encoder = ViTEncoder(device=device)
    encoder.model.eval()
    for p in encoder.model.parameters():
        p.requires_grad = False

    cfg = Config()
    decoder = Decoder(cfg, vision_dim=encoder.vision_dim).to(device)
    # load base pretrained decoder first
    pretrained_path = "../../hw3_data/p2_data/decoder_model.bin"
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=device)
        decoder.load_state_dict(state, strict=False)
        print(f"Loaded pretrained decoder from {pretrained_path}")
    
    # CRITICAL: add LoRA layers with same r/alpha as training before loading adapters
    decoder.add_lora(r=r, alpha=alpha)
    
    # load trained adapters (strict=False to skip missing base params)
    if adapter_path and os.path.exists(adapter_path):
        load_adapters(decoder, adapter_path, device=device)
        print(f"Loaded adapters from {adapter_path}")
    else:
        print(f"Warning: adapter file {adapter_path} not found")
    
    decoder.eval()

    tokenizer = Qwen3Tokenizer(vocab_file, merges_file)
    ds = ImageCaptionDataset(ann_file, images_root, vocab_file, merges_file)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)

    results = {}
    for batch in tqdm(dl, desc="Inference", ncols=120):
        images = batch["images"]
        img_names = batch["img_names"]
        with torch.no_grad():
            vision_feats = encoder.extract(images).to(device)
            start_ids = torch.full((len(images),1), tokenizer.encoder["<|im_start|>"], dtype=torch.long).to(device)
            preds = decoder.generate(
                start_ids,
                max_new_tokens=30,
                eos_token_id=tokenizer.encoder["<|im_end|>"],
                vision_embeds=vision_feats,
                min_new_tokens=5,  # same as training to avoid short collapse
            )
            for i, pid in enumerate(preds):
                # use tokenizer.decode (already handles byte conversion)
                text = tokenizer.decode(pid.cpu().tolist())
                # strip special tokens and normalize
                text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                text = " ".join(text.split())
                # key without extension (required by evaluate.py)
                key = os.path.splitext(img_names[i])[0]
                results[key] = text
    
    with open("pred.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to pred.json ({len(results)} samples)")

if __name__ == "__main__":
    main()
