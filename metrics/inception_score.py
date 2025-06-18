import torch
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from tqdm import tqdm
import gc


def preprocess_images_batch(images, image_size=(299, 299), normalize=False, device='cuda'):
    """
    Preprocess a batch of images with memory optimization.
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
    for img in images:
        processed_images.append(tf(img))
    return torch.stack(processed_images).to(device)


def compute_inception_score_batched(
    gen_loader,
    feature='logits_unbiased',
    splits=10,
    normalize=False,
    dtype=torch.float32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Memory-efficient Inception Score computation using batch processing.

    Args:
        gen_loader: BatchImageLoader for generated images
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

    # Process generated images in batches
    print("Processing generated images for Inception Score...")
    for batch_images, _ in tqdm(gen_loader.batch_generator(), desc="Generated images (IS)"):
        try:
            batch_tensor = preprocess_images_batch(batch_images, normalize=normalize, device=device)
            inception.update(batch_tensor)
            
            # Clear memory
            del batch_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping batch due to error: {e}")
            continue

    print("Computing Inception Score...")
    is_mean, is_std = inception.compute()
    
    # Final cleanup
    del inception
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return is_mean.item(), is_std.item()


def compute_inception_score(
    gen_images,
    feature='logits_unbiased',
    splits=10,
    normalize=False,
    dtype=torch.float32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Legacy Inception Score computation for backward compatibility.
    WARNING: This loads all images into memory at once. Use compute_inception_score_batched for large datasets.

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
    print("WARNING: Using legacy Inception Score computation. Consider using compute_inception_score_batched for better memory efficiency.")
    
    inception = InceptionScore(
        feature=feature,
        splits=splits,
        normalize=normalize
    ).to(device)
    inception.set_dtype(dtype)

    # Process images in smaller chunks to avoid memory issues
    chunk_size = 32  # Reduced chunk size for memory safety
    
    for i in tqdm(range(0, len(gen_images), chunk_size), desc="Processing generated images (IS)"):
        chunk = gen_images[i:i+chunk_size]
        gen_tensor = preprocess_images_batch(chunk, normalize=normalize, device=device)
        inception.update(gen_tensor)
        
        del gen_tensor
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print("Computing Inception Score...")
    is_mean, is_std = inception.compute()
    
    # Final cleanup
    del inception
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return is_mean.item(), is_std.item()
