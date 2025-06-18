import torch
from torchmetrics.image.kid import KernelInceptionDistance
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
            transforms.ToTensor()  # outputs [0,1] float
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # cast to uint8 in [0,255]
        ])

    processed_images = []
    for img in images:
        processed_images.append(tf(img))
    return torch.stack(processed_images).to(device)


def compute_kid_batched(real_loader, 
                       gen_loader,
                       feature_dim=2048, 
                       subsets=100, 
                       subset_size=1000, 
                       normalize=False, 
                       dtype=torch.float32, 
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Memory-efficient KID computation using batch processing.

    Args:
        real_loader: BatchImageLoader for real images
        gen_loader: BatchImageLoader for generated images
        feature_dim (int): Feature layer to use. Default = 2048.
        subsets (int): Number of random subsets.
        subset_size (int): Number of images per subset.
        normalize (bool): If True, assumes inputs are float [0,1]; else uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): cuda or cpu.

    Returns:
        tuple(float, float): KID mean and std.
    """
    # Adjust subset_size based on available data
    min_samples = min(len(real_loader), len(gen_loader))
    if subset_size >= min_samples:
        subset_size = min_samples // 2  # Use half the minimum samples
        print(f"Adjusted subset_size to {subset_size} (half of minimum samples: {min_samples})")
    
    kid_metric = KernelInceptionDistance(
        feature=feature_dim,
        subsets=subsets,
        subset_size=subset_size,
        reset_real_features=True,
        normalize=normalize
    ).to(device)
    kid_metric.set_dtype(dtype)

    # Process real images in batches
    print("Processing real images...")
    for batch_images, _ in tqdm(real_loader.batch_generator(), desc="Real images (KID)"):
        try:
            batch_tensor = preprocess_images_batch(batch_images, normalize=normalize, device=device)
            kid_metric.update(batch_tensor, real=True)
            
            # Clear memory
            del batch_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping real batch due to error: {e}")
            continue

    # Process generated images in batches
    print("Processing generated images...")
    for batch_images, _ in tqdm(gen_loader.batch_generator(), desc="Generated images (KID)"):
        try:
            batch_tensor = preprocess_images_batch(batch_images, normalize=normalize, device=device)
            kid_metric.update(batch_tensor, real=False)
            
            # Clear memory
            del batch_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping generated batch due to error: {e}")
            continue

    print("Computing KID score...")
    kid_mean, kid_std = kid_metric.compute()
    
    # Final cleanup
    del kid_metric
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return kid_mean.item(), kid_std.item()


def compute_kid(real_images, 
                gen_images, 
                feature_dim=2048, 
                subsets=100, 
                subset_size=1000, 
                normalize=False, 
                dtype=torch.float32, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Legacy KID computation function for backward compatibility.
    WARNING: This loads all images into memory at once. Use compute_kid_batched for large datasets.

    Args:
        real_images (List): List of PIL images.
        gen_images (List): List of PIL images.
        feature_dim (int): Feature layer to use. Default = 2048.
        subsets (int): Number of random subsets.
        subset_size (int): Number of images per subset.
        normalize (bool): If True, assumes inputs are float [0,1]; else uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): cuda or cpu.

    Returns:
        tuple(float, float): KID mean and std.
    """
    print("WARNING: Using legacy KID computation. Consider using compute_kid_batched for better memory efficiency.")
    
    # Adjust subset_size to be smaller than the minimum number of samples
    min_samples = min(len(real_images), len(gen_images))
    if subset_size >= min_samples:
        subset_size = min_samples // 2  # Use half the minimum samples
        print(f"Adjusted subset_size to {subset_size} (half of minimum samples: {min_samples})")
    
    kid_metric = KernelInceptionDistance(
        feature=feature_dim,
        subsets=subsets,
        subset_size=subset_size,
        reset_real_features=True,
        normalize=normalize
    ).to(device)
    kid_metric.set_dtype(dtype)

    # Process images in smaller chunks to avoid memory issues
    chunk_size = 32  # Reduced chunk size for memory safety
    
    # Process real images in chunks
    for i in tqdm(range(0, len(real_images), chunk_size), desc="Processing real images (KID)"):
        chunk = real_images[i:i+chunk_size]
        real_tensor = preprocess_images_batch(chunk, normalize=normalize, device=device)
        kid_metric.update(real_tensor, real=True)
        
        del real_tensor
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Process generated images in chunks
    for i in tqdm(range(0, len(gen_images), chunk_size), desc="Processing generated images (KID)"):
        chunk = gen_images[i:i+chunk_size]
        gen_tensor = preprocess_images_batch(chunk, normalize=normalize, device=device)
        kid_metric.update(gen_tensor, real=False)
        
        del gen_tensor
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    print("Computing KID score...")
    kid_mean, kid_std = kid_metric.compute()
    
    # Final cleanup
    del kid_metric
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return kid_mean.item(), kid_std.item()
