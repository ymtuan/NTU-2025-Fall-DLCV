import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Enable VCD sampling BEFORE loading model
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vcd_utils.vcd_add_noise import add_diffusion_noise
from PIL import Image
import torch


def load_model(model_path):
    """Load pretrained LLaVA model"""
    device_map = "auto"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map=device_map
    )
    return tokenizer, model, image_processor


def get_llava_vcd_response(image_path, question, tokenizer, model, image_processor, 
                          cd_alpha=1.0, cd_beta=0.5, noise_step=500):
    """
    Get LLaVA response with Visual Contrastive Decoding
    
    Args:
        image_path: path to image
        question: input question
        tokenizer: tokenizer
        model: LLaVA model
        image_processor: image processor
        cd_alpha: weight for VCD contrastive term (higher = stronger hallucination mitigation)
        cd_beta: threshold for adaptive plausibility constraints
        noise_step: noise step for diffusion (0-999, higher = more noise)
    
    Returns:
        prediction: "Yes" or "No"
    """
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # Add diffusion noise for contrastive decoding (provided by TAs)
    image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=noise_step)
    image_tensor_cd = image_tensor_cd.to(model.device, dtype=torch.float16)
    
    # Prepare prompt
    inp = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    
    # Tokenize
    input_ids = tokenizer_image_token(
        inp, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # Generate response with VCD
    # do_sample=True enables sampling (required for VCD)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            images_cd=image_tensor_cd,
            cd_alpha=cd_alpha,
            cd_beta=cd_beta,
            do_sample=True,
            max_new_tokens=10,
            use_cache=True,
            temperature=0.7  # Control randomness
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


def main(annotation_file, images_root, llava_weight_path, output_file, 
         cd_alpha=1.0, cd_beta=0.5, noise_step=500):
    """Main inference function with VCD"""
    
    # Load model
    print(f"Loading model from {llava_weight_path}...")
    tokenizer, model, image_processor = load_model(llava_weight_path)
    
    print(f"VCD Parameters: alpha={cd_alpha}, beta={cd_beta}, noise_step={noise_step}")
    
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
        
        # Get prediction with VCD
        try:
            pred = get_llava_vcd_response(
                image_path, question, tokenizer, model, image_processor,
                cd_alpha=cd_alpha, cd_beta=cd_beta, noise_step=noise_step
            )
        except Exception as e:
            print(f"Error processing {image_source}: {e}")
            pred = "No"
        
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
    if len(sys.argv) < 5:
        print("Usage: python3 p1_2_vcd_inference.py <annotation_file> <images_root> <llava_weight_path> <output_file> [cd_alpha] [cd_beta] [noise_step]")
        sys.exit(1)
    
    annotation_file = sys.argv[1]
    images_root = sys.argv[2]
    llava_weight_path = sys.argv[3]
    output_file = sys.argv[4]
    
    cd_alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    cd_beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    noise_step = int(sys.argv[7]) if len(sys.argv) > 7 else 500
    
    main(annotation_file, images_root, llava_weight_path, output_file, 
         cd_alpha=cd_alpha, cd_beta=cd_beta, noise_step=noise_step)
    