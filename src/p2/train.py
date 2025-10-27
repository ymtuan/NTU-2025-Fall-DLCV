import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from tqdm import tqdm
from dataset import ImageCaptionDataset, collate_fn
from vit_encoder import ViTEncoder
from decoder_lora import Decoder, Config
from utils import save_adapters, count_trainable_params
from tokenization_qwen3 import Qwen3Tokenizer
from evaluate import CIDERScore, CLIPScore
from torch.utils.data import Subset
import random

# Configs (tweak as needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
images_root = "../../hw3_data/p2_data/images/train"
ann_file = "../../hw3_data/p2_data/train.json"
vocab_file = "vocab.json"
merges_file = "merges.txt"

batch_size = 8
epochs = 15
lr = 3e-4
r = 4
alpha = 16

# add validation files (used to compute best checkpoint)
save_periodic_every = 5  # save extra backup every N epochs
internal_val_frac = 0.05  # fraction of train set to use as internal validation
split_seed = 42  # reproducible split

def freeze_module(m):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_named(model, substrs):
    for n, p in model.named_parameters():
        if any(s in n for s in substrs):
            p.requires_grad = True

def evaluate(decoder, encoder, val_loader, device, pad_id=151643):
    decoder.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        for batch in val_loader:
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            vision_feats = encoder.extract(images).to(device)
            logits = decoder(input_ids=input_ids, vision_embeds=vision_feats)
            v_len = vision_feats.size(1) if vision_feats is not None else 0
            logits_text = logits[:, v_len:, :].contiguous()
            shift_logits = logits_text[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(1, total_batches)

def main():
    # Prepare encoder
    encoder = ViTEncoder(device=device)
    encoder.model.eval()
    freeze_module(encoder.model)

    # Dataset
    ds = ImageCaptionDataset(ann_file, images_root, vocab_file, merges_file)
    # split train -> train / internal-val
    N = len(ds)
    val_size = int(N * internal_val_frac)
    indices = list(range(N))
    random.Random(split_seed).shuffle(indices)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    train_ds = Subset(ds, train_idx)
    val_internal_ds = Subset(ds, val_idx)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dl = DataLoader(val_internal_ds, batch_size=max(1, batch_size // 2), shuffle=False, collate_fn=collate_fn, num_workers=2)

    # tokenizer & metric objects (used for internal validation metrics)
    tokenizer = Qwen3Tokenizer(vocab_file, merges_file)
    cider_evaluator = CIDERScore()
    clip_evaluator = CLIPScore()

    # Decoder
    cfg = Config()
    decoder = Decoder(cfg, vision_dim=encoder.vision_dim).to(device)
    # Load provided pretrained decoder weights (if available)
    pretrained_path = "../../hw3_data/p2_data/decoder_model.bin"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained decoder from {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)
        decoder.load_state_dict(state, strict=False)
    else:
        print(f"Pretrained decoder not found at {pretrained_path}; continuing from random init")

    decoder.add_lora(r=r, alpha=alpha)
    # Freeze base decoder params
    freeze_module(decoder)
    # Unfreeze LoRA adapters and visual_projection
    unfreeze_named(decoder, ["lora_down", "lora_up", "visual_projection"])
    print("trainable params:", count_trainable_params(decoder))

    optim = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad], lr=lr)
    pad_id = 151643
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_val_loss = float("inf")
    best_epoch = -1
    best_cider = -1.0
    best_epoch_by_cider = -1

    for epoch in range(epochs):
        decoder.train()
        epoch_start = time.time()
        running_loss = 0.0
        total_batches = len(dl)
        # wrap dataloader with tqdm for per-batch progress
        pbar = tqdm(dl, total=total_batches, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
        for batch_idx, batch in enumerate(pbar, start=1):
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)  # (B, L)
            # extract vision features
            vision_feats = encoder.extract(images)  # (B, v_len, D)
            logits = decoder(input_ids=input_ids, vision_embeds=vision_feats)
            # When visual prefix is prepended, logits shape = (B, v_len + text_len, V).
            # We should compute loss only over text token positions (ignore visual prefix).
            v_len = vision_feats.size(1) if vision_feats is not None else 0
            logits_text = logits[:, v_len:, :].contiguous()  # (B, text_len, V)
            # causal LM shift: predict next token within text region
            shift_logits = logits_text[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in decoder.parameters() if p.requires_grad], max_norm=1.0)
            optim.step()

            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            cur_lr = optim.param_groups[0]["lr"] if optim.param_groups else lr
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{cur_lr:.2e}")

        pbar.close()

        # epoch summary
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = running_loss / max(1, total_batches)
        tqdm.write(f"Epoch {epoch+1} completed in {epoch_time:.1f}s - avg_loss={epoch_avg_loss:.4f}")

        # Evaluate on internal validation set (loss + CIDEr + CLIPScore)
        # 1) validation loss (fast, uses same evaluate function)
        val_loss = evaluate(decoder, encoder, val_dl, device, pad_id=pad_id)
        tqdm.write(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")

        # 2) generate captions on internal val and compute CIDEr & CLIPScore (may be slower)
        preds = {}
        gts = {}
        # build ground-truth mapping from original ds.anns for indices in val_idx
        for idx in val_idx:
            ann = ds.anns[idx]
            fname = ann.get("file_name") or ann.get("image") or str(ann.get("image_id", idx))
            key = os.path.splitext(fname)[0]
            caps = ann.get("captions") or ann.get("caption") or ann.get("text") or []
            if isinstance(caps, str):
                caps = [caps]
            gts[key] = caps

        decoder.eval()
        with torch.no_grad():
            for batch in val_dl:
                images = batch["images"]
                img_names = batch["img_names"]
                vision_feats = encoder.extract(images).to(device)
                start_ids = torch.full((len(images),1), tokenizer.encoder["<|im_start|>"], dtype=torch.long).to(device)
                preds_ids = decoder.generate(start_ids, max_new_tokens=30, eos_token_id=tokenizer.encoder["<|im_end|>"], vision_embeds=vision_feats)
                for i, pid in enumerate(preds_ids):
                    text = tokenizer.decode(pid.cpu().tolist())
                    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                    key = os.path.splitext(img_names[i])[0]
                    preds[key] = text

        # compute CIDEr & CLIPScore using provided classes
        try:
            cider_score = cider_evaluator(preds, gts)
        except Exception as e:
            cider_score = -1.0
            tqdm.write(f"CIDEr computation failed: {e}")
        try:
            clip_score = clip_evaluator(preds, images_root)
        except Exception as e:
            clip_score = -1.0
            tqdm.write(f"CLIPScore computation failed: {e}")

        tqdm.write(f"Metrics after epoch {epoch+1}: CIDEr={cider_score:.4f} CLIPScore={clip_score:.4f}")

        # save last epoch adapters
        os.makedirs("checkpoints", exist_ok=True)
        save_adapters(decoder, f"checkpoints/adapters_epoch{epoch}.pt")
        save_adapters(decoder, f"checkpoints/adapters_last.pt")
        # select best by CIDEr
        if cider_score > best_cider:
            best_cider = cider_score
            best_epoch_by_cider = epoch
            save_adapters(decoder, "checkpoints/adapters_best_by_cider.pt")
            tqdm.write(f"New best-by-CIDEr adapters saved at epoch {epoch} (CIDEr={cider_score:.4f})")
        # periodic backup
        if (epoch + 1) % save_periodic_every == 0:
            save_adapters(decoder, f"checkpoints/adapters_epoch{epoch}_backup.pt")
            tqdm.write(f"Periodic backup saved for epoch {epoch}")

        tqdm.write(f"Epoch {epoch} finished. best_epoch_by_cider={best_epoch_by_cider} best_cider={best_cider:.4f}")

if __name__ == "__main__":
    main()
