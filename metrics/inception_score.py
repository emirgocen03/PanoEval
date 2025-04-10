import torch
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from tqdm import tqdm


def preprocess_images(images, image_size=(299, 299), normalize=False):
    """
    Preprocess images to match the input requirements for the metric.
    Returns a tensor of shape (N, 3, H, W).
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

    processed_images = []
    for img in tqdm(images, desc="Preprocessing (Inception score)"):
        processed_images.append(tf(img))
    return torch.stack(processed_images)


def compute_inception_score(
    gen_images,
    feature='logits_unbiased',
    splits=10,
    normalize=False,
    dtype=torch.float64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute Inception Score for generated images.

    Args:
        gen_images (List): List of PIL images.
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

    # Preprocess images
    gen_imgs = preprocess_images(gen_images, normalize=normalize).to(device)

    # Update and compute
    inception.update(gen_imgs)
    is_mean, is_std = inception.compute()
    return is_mean.item(), is_std.item()
