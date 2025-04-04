import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def load_images_tensor(folder, image_size=(512, 256), device='cuda'):
    """
    Loads and resizes all images in a folder. Returns a tensor of shape (N, 3, H, W).
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    tensors = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            tensors.append(tf(img).to(device))
    return tensors


def scharr_kernel():
    """3x3 horizontal Scharr kernel as tensor (second-order approximation as in OpenCV)"""
    return torch.tensor([
        [3., 0., -3.],
        [10., 0., -10.],
        [3., 0., -3.]
    ]).view(1, 1, 3, 3) / 16.0


def extract_seam_region(image_tensor, seam_width=6):
    """
    Extracts a (3, H, seam_width) tensor from both sides of the panorama.
    Returns a seam tensor of shape (3, H, seam_width * 2).
    """
    left = image_tensor[:, :, :seam_width]
    right = image_tensor[:, :, -seam_width:]
    return torch.cat([right, left], dim=2)  # concat horizontally


def compute_ds_score(gray_seam, kernel, eps=0.1):
    """
    Compute the Discontinuity Score (DS) for a seam region using horizontal Scharr edge detection.

    Args:
        gray_seam (Tensor): Tensor of shape (1, 1, H, 2 * seam_width) (e.g., [1, 1, H, 12])
        kernel (Tensor): Horizontal Scharr kernel
        eps (float): Stability constant to prevent divide-by-zero

    Returns:
        float: Scalar DS score for this seam
    """
    L = gray_seam.shape[2]  # Height of image
    convolved = F.conv2d(gray_seam, kernel, padding=1)  # (1, 1, H, 2 * seam_width)
    conv_abs = torch.abs(convolved.squeeze(0).squeeze(0))  # (H, 2 * seam_width)

    top = conv_abs[:, 2] / (conv_abs[:, 1] + eps)
    bottom = conv_abs[:, 3] / (conv_abs[:, 4] + eps)
    return (top.sum() + bottom.sum()).item() / (2 * L)


def compute_discontinuity_score(folder, image_size=(512, 256), device='cuda'):
    """
    Computes average Discontinuity Score for all generated panoramas in folder.
    """
    kernel = scharr_kernel().to(device)
    seam_width = 6
    height = image_size[1]
    scale_factor = seam_width / height

    image_tensors = load_images_tensor(folder, image_size=image_size, device=device)

    scores = []
    for img_tensor in image_tensors:
        seam = extract_seam_region(img_tensor, seam_width=seam_width)
        gray = 0.2989 * seam[0] + 0.5870 * seam[1] + 0.1140 * seam[2]  # grayscale
        gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W=6*2)

        score = compute_ds_score(gray, kernel)
        scores.append(score * scale_factor)

    return sum(scores) / len(scores)
