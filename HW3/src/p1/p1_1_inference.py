import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import torch

def _normalize_yes_no(text):
    if text is None:
        return "no"
    import re
    tokens = re.findall(r"[a-zA-Z]+", text.strip().lower())
    if not tokens:
        return "no"
    first = tokens[0]
    if first in ("yes", "y"):
        return "yes"
    if first in ("no", "n"):
        return "no"
    return "no" 

# Load pretrained LLaVA model
def load_model(model_path):
    device_map = "auto"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map=device_map
    )
    model.eval()
    return tokenizer, model, image_processor

def get_llava_response(image_path, question, tokenizer, model, image_processor):
    """
    Get LLaVA model response for a yes/no question
    Returns: "yes" or "no"
    """
    # Load and process image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Image load error: {e}")
        return "no"
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # Use proper LLaVA conversation template
    conv = conv_templates["llava_v1"].copy()
    # Add instruction to answer strictly with yes/no
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\nAnswer the question with a single yes/no.\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,      # greedy - standard baseline
            max_new_tokens=1,
            use_cache=True
        )
    
    # Extract only the generated part (remove input tokens)
    input_len = input_ids.shape[1]
    generated_ids = output_ids[:, input_len:]
    
    # If no tokens were generated, return "no"
    if generated_ids.shape[1] == 0:
        return "no"
    
    # Decode response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    normalized = _normalize_yes_no(response)
    return normalized  # Return only the normalized response

def main(annotation_file, images_root, llava_weight_path, output_file):
    """Main inference function"""
    
    # Load model
    print(f"Loading model from {llava_weight_path}...")
    tokenizer, model, image_processor = load_model(llava_weight_path)
    
    # Load annotation data
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file) as f:
        data = json.load(f)
    
    results = []
    
    for i, item in enumerate(data):
        image_source = item['image_source']
        question = item['question']
        qid = item.get("question_id", i)
        
        # Construct image path
        image_path = os.path.join(images_root, f"{image_source}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(images_root, f"{image_source}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {image_source}")
            continue
        
        # Get prediction
        pred = get_llava_response(image_path, question, tokenizer, model, image_processor)
        
        results.append({
            "image_source": image_source,
            "question": question,
            "predict": pred
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)}")
    
    # Save output (manual pretty list with one object per line)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for idx, obj in enumerate(results):
            f.write('  {\n')
            f.write(f'    "image_source": "{obj["image_source"]}",\n')
            f.write(f'    "question": "{obj["question"]}",\n')
            f.write(f'    "predict": "{obj["predict"]}"\n')
            if idx < len(results) - 1:
                f.write('  },\n')
            else:
                f.write('  }\n')
        f.write(']\n')
    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 p1_1_inference.py <annotation_file> <images_root> <llava_weight_path> <output_file>")
        sys.exit(1)
    
    annotation_file = sys.argv[1]
    images_root = sys.argv[2]
    llava_weight_path = sys.argv[3]
    output_file = sys.argv[4]
    
    main(annotation_file, images_root, llava_weight_path, output_file)