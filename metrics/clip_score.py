import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore


def load_images_tensor(folder, image_size=(224, 224), normalize=True):
    """
    Load and preprocess images to match CLIP input requirements.
    Returns list of (C, H, W) tensors.
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # float32 in [0,1]
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),  # CLIP normalization
    ])

    images = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(tf(img))
    return images


def load_captions(text_file, image_list):
    """
    Load text prompts or captions. Expected format: one caption per line matching order of images.
    """
    with open(text_file, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f.readlines()]
    if len(captions) != len(image_list):
        raise ValueError(f"Mismatch: {len(captions)} captions vs {len(image_list)} images")
    return captions


def compute_clip_score(
    gen_dir,
    text_file,
    model_name="openai/clip-vit-large-patch14",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute CLIP Score between generated images and their corresponding captions.

    Args:
        gen_dir (str): Path to generated images.
        text_file (str): Path to file with one caption per image (in filename order).
        model_name (str): CLIP model variant.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average CLIP Score [0, 100].
    """
    # Load images and captions
    gen_imgs = load_images_tensor(gen_dir)
    captions = load_captions(text_file, gen_imgs)

    # Convert to batch
    image_tensor = torch.stack(gen_imgs).to(device)

    # Initialize CLIPScore metric
    metric = CLIPScore(model_name_or_path=model_name).to(device)

    # Compute score
    metric.update(image_tensor, captions)
    score = metric.compute()
    return score.item()
