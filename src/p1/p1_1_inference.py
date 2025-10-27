import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
import torch

# Load pretrained LLaVA model
def load_model(model_path):
    device_map = "auto"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map=device_map
    )
    return tokenizer, model, image_processor

def get_llava_response(image_path, question, tokenizer, model, image_processor):
    """
    Get LLaVA model response for a yes/no question
    Returns: "Yes" or "No"
    """
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # Prepare prompt
    inp = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    
    # Tokenize
    input_ids = tokenizer_image_token(
        inp, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=10,
            use_cache=True
        )
    
    # Extract only the generated part (remove input tokens)
    input_len = input_ids.shape[1]
    generated_ids = output_ids[:, input_len:]
    
    # Decode response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Parse response to "Yes" or "No"
    response_lower = response.lower()
    if "yes" in response_lower:
        return "Yes"
    else:
        return "No"

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
            "image_id": image_source,
            "question_id": i,
            "text": pred
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)}")
    
    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
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