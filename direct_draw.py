import os
import json
import re
import cv2
import argparse

# Import your drawing utility functions
from utils.annotation_utils import parse_visual_prompts, AnnotationRenderer

def process_format_1_model_output(item, item_index, dataset_root, output_dir):
    """
    Process Format 1 (Model Output Format):
    - Parses text instructions from the <Answer> block.
    - Requires coordinate scaling from the normalized 0-1000 system.
    """
    # Find the assistant's reply
    assistant_msg = next((msg["content"] for msg in item.get("messages", []) if msg["role"] == "assistant"), None)
    if not assistant_msg:
        print(f"Data [{item_index}] Format 1: Assistant reply not found, skipping.")
        return

    # Extract the <Answer> block via regex
    answer_match = re.search(r'<Answer>(.*?)</Answer>', assistant_msg, re.DOTALL)
    if not answer_match:
        print(f"Data [{item_index}] Format 1: <Answer> tag not found, skipping.")
        return
        
    answer_content = answer_match.group(1).strip()
    
    # Extract the keyframe
    keyframe_match = re.search(r'"keyframe":\s*"(\d+)"', answer_content)
    keyframe_idx = int(keyframe_match.group(1)) if keyframe_match else None
    
    # Extract visual prompts (handle potential multiline escapes)
    vp_match = re.search(r'"visual prompts":\s*"(.*?)"', answer_content, re.DOTALL)
    if not vp_match or not keyframe_idx:
        print(f"Data [{item_index}] Format 1: Missing keyframe or visual prompts info, skipping.")
        return

    visual_prompts_str = vp_match.group(1)
    
    # Get image path
    images_list = item.get("images", [])
    if not (0 < keyframe_idx <= len(images_list)):
        print(f"Data [{item_index}] Format 1: Keyframe index {keyframe_idx} out of bounds, skipping.")
        return
        
    rel_img_path = images_list[keyframe_idx - 1]
    abs_img_path = os.path.join(dataset_root, rel_img_path)
    
    if not os.path.exists(abs_img_path):
        print(f"Image not found: {abs_img_path}")
        return

    # Parse drawing commands (list of dictionaries)
    render_commands = parse_visual_prompts(visual_prompts_str)
    
    # Load image and draw (Format 1 forces normalized coordinate scaling to True)
    img = cv2.imread(abs_img_path)
    renderer = AnnotationRenderer(img, use_normalized_coordinates=True)
    renderer.draw_commands(render_commands)
    
    save_path = os.path.join(output_dir, f"task_{item_index}_format1_keyframe_{keyframe_idx}.jpg")
    cv2.imwrite(save_path, renderer.image)
    print(f"Format 1 visualization saved to: {save_path}")

def process_format_2_original_annotation(item, item_index, dataset_root, output_dir):
    """
    Process Format 2 (Original Annotation Data Format):
    - Reads 'avoid_keyframe_annotations_code' or 'correction' fields directly from JSON.
    - Uses absolute pixel coordinates on the original image, NO scaling needed.
    """
    keyframe_info = item.get("keyframe", [{}])[0]
    
    # Iterate through Avoidance and Correction phases
    for phase, prefix in [("avoidance", "avoid"), ("correction", "correct")]:
        rel_img_path = keyframe_info.get(f"{prefix}_keyframe")
        
        # Get drawing commands for the corresponding phase
        phase_data = item.get(phase, [{}])[0]
        render_commands = phase_data.get(f"{prefix}_keyframe_annotations_code", [])
        
        if rel_img_path and render_commands:
            abs_img_path = os.path.join(dataset_root, rel_img_path)
            
            if not os.path.exists(abs_img_path):
                print(f"Image not found: {abs_img_path}")
                continue

            # Compatibility fix: older formats used 'start' and 'end' for rotate_arrow, newer uses 'center'
            for cmd in render_commands:
                if cmd.get('type') == 'rotate_arrow' and 'center' not in cmd:
                    start = cmd.get('start')
                    end = cmd.get('end')
                    if start and end:
                        cmd['center'] = [int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2)]

            # Load image and draw (Format 2 forces normalized coordinate scaling to False)
            img = cv2.imread(abs_img_path)
            renderer = AnnotationRenderer(img, use_normalized_coordinates=False)
            renderer.draw_commands(render_commands)
            
            save_path = os.path.join(output_dir, f"task_{item_index}_format2_{phase}.jpg")
            cv2.imwrite(save_path, renderer.image)
            print(f"Format 2 ({phase}) visualization saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Directly draw annotations from a JSON file onto images.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON data file.")
    parser.add_argument("--dataset_root", type=str, default="./ViFailback-Dataset", help="Root directory of the dataset images.")
    parser.add_argument("--output_dir", type=str, default="./visualized_results", help="Directory to save the visualized results.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading JSON data: {args.json_path}")
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    if not isinstance(data_list, list):
        data_list = [data_list] # Handle single dictionary case gracefully

    for i, item in enumerate(data_list):
        print(f"\n--- Processing item {i+1} ---")
        
        # Simple format detection strategy:
        if "messages" in item:
            # Format 1: Contains conversational messages
            process_format_1_model_output(item, i + 1, args.dataset_root, args.output_dir)
            
        elif "avoidance" in item or "correction" in item:
            # Format 2: Contains avoidance or correction direct command fields
            process_format_2_original_annotation(item, i + 1, args.dataset_root, args.output_dir)
            
        else:
            print(f"Failed to recognize data structure format for item [{i+1}].")

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()