import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from panfusion_faed import FrechetAutoEncoderDistance


def load_images_tensor(folder, image_size=(512, 256), device='cuda', normalize=False):
    """
    Load and preprocess panorama images for FAED.
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # float32 in [0,1]
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8) if not normalize else x)
    ])
    
    images = []
    for fname in tqdm(os.listdir(folder), desc=f"Loading images from {folder}"):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(tf(img))
    return torch.stack(images).to(device)


def compute_faed(
    real_dir,
    gen_dir,
    pano_height=256,
    image_size=(512, 256),
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute Frechet AutoEncoder Distance (FAED) between real and generated panoramic images.

    Args:
        real_dir (str): Path to real images.
        gen_dir (str): Path to generated images.
        pano_height (int): Used for estimating feature vector size.
        image_size (tuple): Resize target (W, H).
        device (str): cuda or cpu

    Returns:
        float: FAED score
    """
    # Initialize metric
    metric = FrechetAutoEncoderDistance(pano_height=pano_height).to(device)

    # Load and process images
    real_imgs = load_images_tensor(real_dir, image_size=image_size, device=device)
    gen_imgs = load_images_tensor(gen_dir, image_size=image_size, device=device)

    # Update metric
    metric.update(real_imgs, real=True)
    metric.update(gen_imgs, real=False)

    # Compute FAED
    faed_score = metric.compute().item()
    return faed_score
