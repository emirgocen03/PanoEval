import os
import torch
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from PIL import Image


def load_images_tensor(folder, image_size=(299, 299), normalize=False):
    """
    Load all images from a folder and return a torch.Tensor of shape (N, 3, H, W).
    """
    if normalize:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()  # float [0,1]
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # uint8 [0,255]
        ])

    tensors = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            tensors.append(tf(img))
    return torch.stack(tensors)


def compute_inception_score(
    gen_dir,
    feature='logits_unbiased',
    splits=10,
    normalize=False,
    dtype=torch.float64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute Inception Score for generated images.

    Args:
        gen_dir (str): Path to generated images.
        feature (str): Feature layer of InceptionV3. Default: 'logits_unbiased'.
        splits (int): Number of splits to estimate std.
        normalize (bool): True if images are [0,1] float, False if uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): Device to run the metric on.

    Returns:
        (float, float): Tuple of (IS mean, IS std)
    """
    inception = InceptionScore(
        feature=feature,
        splits=splits,
        normalize=normalize
    ).to(device)
    inception.set_dtype(dtype)

    # Load generated images
    gen_imgs = load_images_tensor(gen_dir, normalize=normalize).to(device)

    # Update and compute
    inception.update(gen_imgs)
    is_mean, is_std = inception.compute()
    return is_mean.item(), is_std.item()
