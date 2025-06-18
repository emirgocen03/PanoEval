import torch
from torchvision import transforms
from .panfusion_faed import FrechetAutoEncoderDistance
from tqdm import tqdm


def preprocess_images(images, image_size=(512, 256), normalize=False):
    """
    Preprocess panorama images for FAED.
    Returns a tensor of shape (N, 3, H, W).
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # float32 in [0,1]
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8) if not normalize else x)
    ])
    
    preprocessed_images = []
    for img in tqdm(images, desc="Preprocessing (FAED)"):
        preprocessed_images.append(tf(img))
    return torch.stack(preprocessed_images)


def compute_faed(
    real_images,
    gen_images,
    pano_height=512,
    image_size=(512, 256),
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute Frechet AutoEncoder Distance (FAED) between real and generated panoramic images.

    Args:
        real_images (List): List of PIL images.
        gen_images (List): List of PIL images.
        pano_height (int): Used for estimating feature vector size.
        image_size (tuple): Resize target (W, H).
        device (str): cuda or cpu

    Returns:
        float: FAED score
    """
    # Initialize metric
    metric = FrechetAutoEncoderDistance(pano_height=pano_height).to(device)

    # Preprocess images
    real_imgs = preprocess_images(real_images, image_size=image_size).to(device)
    gen_imgs = preprocess_images(gen_images, image_size=image_size).to(device)

    # Update metric
    metric.update(real_imgs, real=True)
    metric.update(gen_imgs, real=False)

    # Compute FAED
    print("Calculating...")
    faed_score = metric.compute().item()
    return faed_score
