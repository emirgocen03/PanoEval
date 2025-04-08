from PIL import Image
import os
from tqdm import tqdm

def load_images(folder):
    """
    Load all images from a folder.
    Returns: list of PIL images
    """
    images = []
    for fname in tqdm(sorted(os.listdir(folder)), desc=f"Loading images from {folder}"):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images


def load_text_prompts(folder):
    """
    Load all text prompts from a folder.
    Returns: list of text prompts
    """
    prompts = []
    for fname in tqdm(sorted(os.listdir(folder)), desc=f"Loading text prompts from {folder}"):
        if fname.lower().endswith((".txt")):
            prompt_path = os.path.join(folder, fname)
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts.append(f.read().strip())