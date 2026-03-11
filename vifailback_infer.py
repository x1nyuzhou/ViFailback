import os
import json
import re
import cv2
import torch
import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import custom parser and built-in normalized renderer
from utils.annotation_utils import parse_visual_prompts, AnnotationRenderer

def parse_model_output(output_text):
    """
    Extracts content inside the <Answer> tags, and parses out the keyframe
    and visual prompts.
    """
    answer_match = re.search(r'<Answer>(.*?)</Answer>', output_text, re.DOTALL)
    if not answer_match:
        return None, None
        
    answer_content = answer_match.group(1).strip()
    keyframe_match = re.search(r'"keyframe":\s*"(\d+)"', answer_content)
    keyframe_idx = int(keyframe_match.group(1)) if keyframe_match else None
    
    vp_match = re.search(r'"visual prompts":\s*"(.*?)"', answer_content, re.DOTALL)
    if vp_match:
        # Decode escape characters to get clean commands
        visual_prompts_str = bytes(vp_match.group(1), "utf-8").decode("unicode_escape", "ignore")
    else:
        visual_prompts_str = None
    
    return keyframe_idx, visual_prompts_str

def load_model(model_path):
    print(f"Loading vision-language model from: {model_path} ...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model successfully loaded!")
    return model, processor

def run_qwen_inference(model, processor, text_prompt, image_paths):
    content = [{"type": "image", "image": img_path} for img_path in image_paths]
    content.append({"type": "text", "text": text_prompt})
    
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def main():
    parser = argparse.ArgumentParser(description="Run visual-language model inference and draw generated visual prompts.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (e.g., LLaMA-Factory output dir).")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON data file containing inference tasks.")
    parser.add_argument("--dataset_root", type=str, default="./ViFailback_Dataset", help="Root directory of the dataset images.")
    parser.add_argument("--output_dir", type=str, default="./visualized_results", help="Directory to save the visualized results.")
    parser.add_argument("--disable_normalization", action="store_true", help="Pass this flag if the model outputs absolute coordinates instead of 0-1000.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    use_normalized_coords = not args.disable_normalization

    # 1. Load model
    model, processor = load_model(args.model_path)

    # 2. Read data
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 3. Process loop
    for i, data_item in enumerate(data_list):
        print(f"\n--- Processing item {i+1} ---")
        
        rel_image_paths = data_item.get("images", [])
        abs_image_paths = [os.path.join(args.dataset_root, p) for p in rel_image_paths]
        raw_prompt = data_item["messages"][0]["content"]
        clean_prompt = raw_prompt.replace("<image>", "").strip()

        # Inference phase
        print("Waiting for model inference...")
        model_output = run_qwen_inference(model, processor, clean_prompt, abs_image_paths)
        print(f"Model Output:\n{model_output}\n")
        
        # Extract target frame and text instructions
        keyframe_num, vp_str = parse_model_output(model_output)
        
        if keyframe_num is None or vp_str is None:
            print("Failed to parse output, skipping this item.")
            continue
            
        print(f"Parsed Keyframe: {keyframe_num}")
        print(f"Parsed Visual Prompt: {vp_str}")
        
        # Drawing phase
        target_img_idx = keyframe_num - 1
        
        if 0 <= target_img_idx < len(abs_image_paths):
            target_img_path = abs_image_paths[target_img_idx]
            if os.path.exists(target_img_path):
                img = cv2.imread(target_img_path)
                
                # Directly obtain the parsed coordinate instructions list
                render_commands = parse_visual_prompts(vp_str)
                
                # Render the drawings, passing the normalized toggle variable
                renderer = AnnotationRenderer(img, use_normalized_coordinates=use_normalized_coords)
                renderer.draw_commands(render_commands)
                
                # Save the image
                save_name = f"task_{i+1}_keyframe_{keyframe_num}_rendered.jpg"
                save_path = os.path.join(args.output_dir, save_name)
                cv2.imwrite(save_path, renderer.image)
                print(f"Visualization result saved to: {save_path}")
            else:
                print(f"Image not found: {target_img_path}")
        else:
            print(f"Keyframe index out of bounds: {keyframe_num}")

if __name__ == "__main__":
    main()