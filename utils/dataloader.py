import json
from PIL import Image
import os
from tqdm import tqdm
import gc
import torch


class BatchImageLoader:
    """Memory-efficient batch image loader for large datasets"""
    
    def __init__(self, jsonl_path, batch_size=32, max_images=None):
        self.jsonl_path = jsonl_path
        self.batch_size = batch_size
        self.max_images = max_images
        self._count_entries()
    
    def _count_entries(self):
        """Count total entries in JSONL file"""
        self.total_entries = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for _ in f:
                self.total_entries += 1
                if self.max_images and self.total_entries >= self.max_images:
                    break
    
    def __len__(self):
        return min(self.total_entries, self.max_images or self.total_entries)
    
    def batch_generator(self):
        """Generator that yields batches of images"""
        batch_images = []
        batch_prompts = []
        count = 0
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading {self.jsonl_path}", total=len(self)):
                if self.max_images and count >= self.max_images:
                    break
                    
                entry = json.loads(line)
                
                try:
                    # Load and process image
                    img_path = entry.get('gen_img_path') or entry.get('real_img_path')
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    batch_prompts.append(entry.get('gen_image_prompt', None))
                    
                    # Yield batch when full
                    if len(batch_images) >= self.batch_size:
                        yield batch_images, batch_prompts
                        batch_images.clear()
                        batch_prompts.clear()
                        gc.collect()  # Force garbage collection
                        
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    continue
                
                count += 1
            
            # Yield remaining images
            if batch_images:
                yield batch_images, batch_prompts


def load_generated_jsonl(jsonl_path, batch_size=32, max_images=None):
    """
    Memory-efficient loading of generated images and prompts from a JSONL file.
    Returns: BatchImageLoader for streaming access
    """
    return BatchImageLoader(jsonl_path, batch_size, max_images)


def load_real_jsonl(jsonl_path, batch_size=32, max_images=None):
    """
    Memory-efficient loading of real images from a JSONL file.
    Returns: BatchImageLoader for streaming access
    """
    return BatchImageLoader(jsonl_path, batch_size, max_images)


def load_all_images_legacy(loader):
    """
    Legacy function to load all images at once (for backward compatibility)
    WARNING: Use only for small datasets due to memory constraints
    """
    all_images = []
    all_prompts = []
    
    for batch_images, batch_prompts in loader.batch_generator():
        all_images.extend(batch_images)
        all_prompts.extend(batch_prompts)
    
    return all_images, all_prompts


def convert_dirs_to_jsonl(gen_dir, prompt_dir, real_dir, gen_jsonl_path, real_jsonl_path):
    """
    Utility to convert directories to JSONL format for generated and real images.
    - gen_dir: directory with generated images
    - prompt_dir: directory with text prompts (filenames must match images, except extension)
    - real_dir: directory with real images
    - gen_jsonl_path: output path for generated.jsonl
    - real_jsonl_path: output path for real.jsonl
    """
    # Write generated.jsonl
    with open(gen_jsonl_path, 'w', encoding='utf-8') as gen_out:
        for fname in sorted(os.listdir(gen_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(gen_dir, fname)
                prompt_path = os.path.join(prompt_dir, os.path.splitext(fname)[0] + ".txt") if prompt_dir else None
                prompt = None
                if prompt_path and os.path.exists(prompt_path):
                    with open(prompt_path, 'r', encoding='utf-8') as pf:
                        prompt = pf.read().strip()
                entry = {"gen_img_path": img_path, "gen_image_prompt": prompt}
                gen_out.write(json.dumps(entry) + "\n")
    
    # Write real.jsonl
    with open(real_jsonl_path, 'w', encoding='utf-8') as real_out:
        for fname in sorted(os.listdir(real_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(real_dir, fname)
                entry = {"real_img_path": img_path}
                real_out.write(json.dumps(entry) + "\n")


def convert_matterport_to_jsonl(captions_dir, eqr_dir, gen_jsonl_path, real_jsonl_path):
    """
    Convert Matterport dataset to JSONL format.
    - captions_dir: directory containing caption files organized by scene ID
    - eqr_dir: directory containing equirectangular images organized by scene ID
    - gen_jsonl_path: output path for generated.jsonl (will contain captions as prompts)
    - real_jsonl_path: output path for real.jsonl (will contain equirectangular images)
    """
    # Write generated.jsonl (using captions as prompts)
    with open(gen_jsonl_path, 'w', encoding='utf-8') as gen_out:
        for scene_id in tqdm(sorted(os.listdir(captions_dir)), desc="Processing scenes for generated.jsonl"):
            scene_caption_dir = os.path.join(captions_dir, scene_id)
            if not os.path.isdir(scene_caption_dir):
                continue
                
            for caption_file in sorted(os.listdir(scene_caption_dir)):
                if not caption_file.endswith('_caption.txt'):
                    continue
                    
                # Get image ID from caption filename
                img_id = caption_file.replace('_caption.txt', '')
                
                # Read caption
                caption_path = os.path.join(scene_caption_dir, caption_file)
                with open(caption_path, 'r', encoding='utf-8') as cf:
                    caption = cf.read().strip()
                
                # Get corresponding equirectangular image path
                eqr_img_name = f"{img_id}_equirectangular.jpg"
                eqr_img_path = os.path.join(eqr_dir, scene_id, eqr_img_name)
                
                if os.path.exists(eqr_img_path):
                    entry = {
                        "gen_img_path": eqr_img_path,
                        "gen_image_prompt": caption
                    }
                    gen_out.write(json.dumps(entry) + "\n")
    
    # Write real.jsonl (using equirectangular images)
    with open(real_jsonl_path, 'w', encoding='utf-8') as real_out:
        for scene_id in tqdm(sorted(os.listdir(eqr_dir)), desc="Processing scenes for real.jsonl"):
            scene_eqr_dir = os.path.join(eqr_dir, scene_id)
            if not os.path.isdir(scene_eqr_dir):
                continue
                
            for eqr_file in sorted(os.listdir(scene_eqr_dir)):
                if not eqr_file.endswith('_equirectangular.jpg'):
                    continue
                    
                eqr_img_path = os.path.join(scene_eqr_dir, eqr_file)
                entry = {"real_img_path": eqr_img_path}
                real_out.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    convert_matterport_to_jsonl(
        captions_dir="/home/egocen21/datasets/matterport_captions",
        eqr_dir="/home/egocen21/datasets/matterport_eqr",
        gen_jsonl_path="/home/egocen21/datasets/matterport_generated.jsonl",
        real_jsonl_path="/home/egocen21/datasets/matterport_real.jsonl"
    )