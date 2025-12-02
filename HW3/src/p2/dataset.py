import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from tokenization_qwen3 import Qwen3Tokenizer

PAD_ID = 151643
START = "<|im_start|>"
END = "<|im_end|>"

def _resolve_image_name_by_id(images_root: str, image_id) -> str:
    """
    Try common filename patterns for numeric image_id: '123.jpg', '123.png', ...
    Return first existing filename (basename) or str(image_id) as fallback.
    """
    base = str(image_id)
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        cand = base + ext
        if os.path.exists(os.path.join(images_root, cand)):
            return cand
    # try zero-padded variants up to 6 digits
    for width in (3,4,5,6):
        cand = base.zfill(width) + ".jpg"
        if os.path.exists(os.path.join(images_root, cand)):
            return cand
    return base  # fallback, may be a full filename already

class ImageCaptionDataset(Dataset):
    """
    Expects annotation json in several possible formats:
      - list of records [{...}, ...]
      - COCO-style dict with 'images' and 'annotations'
      - dict mapping image_name -> caption (or -> record)
    Normalizes to a list of records with fields 'file_name' and 'captions'.
    """
    def __init__(self, ann_file, images_root, tokenizer_vocab, tokenizer_merges, transform=None, is_inference=False):
        self.is_inference = is_inference
        if not is_inference and ann_file:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Normalize different json layouts to a list of records
            if isinstance(data, dict):
                if "annotations" in data and "images" in data:  # COCO-like with images list
                    images_by_id = {img.get("id"): img for img in data.get("images", [])}
                    anns = []
                    for ann in data.get("annotations", []):
                        img_entry = images_by_id.get(ann.get("image_id"))
                        fname = None
                        if img_entry:
                            fname = img_entry.get("file_name") or img_entry.get("file_name")
                        fname = fname or ann.get("file_name") or ann.get("image") or ann.get("image_id")
                        caps = ann.get("caption") or ann.get("captions") or ann.get("text") or ann.get("sentences")
                        if caps is None:
                            caps = [ann.get("caption","")]
                        elif isinstance(caps, str):
                            caps = [caps]
                        anns.append({"file_name": fname, "captions": caps})
                    self.anns = anns
                elif "annotations" in data and "images" not in data:
                    # simple annotations list with image_id and caption entries (your provided format)
                    anns = []
                    for ann in data.get("annotations", []):
                        img_id = ann.get("image_id") if ann.get("image_id") is not None else ann.get("id")
                        fname = _resolve_image_name_by_id(images_root, img_id)
                        caps = ann.get("caption") or ann.get("captions") or ann.get("text") or ""
                        if isinstance(caps, str):
                            caps = [caps]
                        anns.append({"file_name": fname, "captions": caps})
                    self.anns = anns
                else:
                    # dict mapping or single record dict
                    vals = list(data.values())
                    # if mapping image_name -> caption string
                    if all(isinstance(v, str) for v in vals):
                        self.anns = [{"file_name": k, "captions":[v]} for k,v in data.items()]
                    else:
                        # treat values as records
                        self.anns = vals
            elif isinstance(data, list):
                self.anns = data
            else:
                # fallback wrap
                self.anns = [data]

            self.images_root = images_root
            self.transform = transform
            self.tokenizer = Qwen3Tokenizer(tokenizer_vocab, tokenizer_merges)
            self.ids = list(range(len(self.anns)))
        else:
            # For inference, just list all images in the folder
            self.image_files = sorted([f for f in os.listdir(images_root) if f.endswith(('.jpg', '.png'))])
            self.images_root = images_root
            self.transform = transform
            self.tokenizer = Qwen3Tokenizer(tokenizer_vocab, tokenizer_merges)
            self.ids = list(range(len(self.image_files)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.is_inference:
            img_name = self.image_files[idx]
            caption = ""
        else:
            rec = self.anns[idx]

            # If rec is a simple string -> filename
            if isinstance(rec, str):
                img_name = rec
                caption = ""
            elif isinstance(rec, dict):
                # image path heuristics
                img_key = None
                for k in ("file_name","image","image_id","filename","img"):
                    if k in rec:
                        img_key = k
                        break
                img_name = rec.get(img_key) if img_key else rec.get("id") or rec.get("img") or rec.get("file_name")
                # captions can be under several keys
                caption = rec.get("captions") or rec.get("caption") or rec.get("text") or rec.get("sentences")
                if isinstance(caption, list):
                    caption = caption[0] if len(caption)>0 else ""
                if caption is None:
                    caption = ""
            else:
                img_name = str(rec)
                caption = ""

        img_path = os.path.join(self.images_root, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if not self.is_inference:
            # add start/end tokens and tokenize
            text = f"{START} {caption} {END}"
            input_ids = self.tokenizer.encode(text)
            return {"image": image, "input_ids": torch.LongTensor(input_ids), "img_name": img_name}
        else:
            return {"image": image, "img_name": img_name}

def collate_fn(batch):
    images = [item["image"] for item in batch]
    img_names = [item["img_name"] for item in batch]
    
    # Check if input_ids exist (training mode) or not (inference mode)
    if "input_ids" in batch[0]:
        input_id_lists = [item["input_ids"] for item in batch]
        max_len = max([l.size(0) for l in input_id_lists])
        padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
        for i, t in enumerate(input_id_lists):
            padded[i, : t.size(0)] = t
        return {"images": images, "input_ids": padded, "img_names": img_names}
    else:
        # Inference mode: no input_ids
        return {"images": images, "img_names": img_names}
