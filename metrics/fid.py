import torch
from torchmetrics.image.fid import FrechetInceptionDistance
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
        processed_img = tf(img)
        processed_images.append(processed_img)
    
    batch_tensor = torch.stack(processed_images)
    return batch_tensor.to(device)


def compute_fid_batched(real_loader, gen_loader, 
                       feature_dim=2048, 
                       normalize=False, 
                       dtype=torch.float64, 
                       device='cuda' if torch.cuda.is_available() else 'cpu',
                       batch_size=32):
    """
    Memory-efficient FID computation using batch processing.
    
    Args:
        real_loader: BatchImageLoader for real images
        gen_loader: BatchImageLoader for generated images  
        feature_dim (int): InceptionV3 feature layer to use. Default = 2048.
        normalize (bool): If True, assumes inputs are float [0,1]; else uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): cuda or cpu.
        batch_size (int): Batch size for processing.

    Returns:
        float: FID score.
    """
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(
        feature=feature_dim,
        reset_real_features=True,
        normalize=normalize
    ).to(device)
    fid_metric.set_dtype(dtype)

    # Process real images in batches
    print("Processing real images...")
    for batch_images, _ in tqdm(real_loader.batch_generator(), desc="Real images"):
        try:
            batch_tensor = preprocess_images_batch(batch_images, normalize=normalize, device=device)
            fid_metric.update(batch_tensor, real=True)
            
            # Clear memory
            del batch_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping batch due to error: {e}")
            continue

    # Process generated images in batches  
    print("Processing generated images...")
    for batch_images, _ in tqdm(gen_loader.batch_generator(), desc="Generated images"):
        try:
            batch_tensor = preprocess_images_batch(batch_images, normalize=normalize, device=device)
            fid_metric.update(batch_tensor, real=False)
            
            # Clear memory
            del batch_tensor
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping batch due to error: {e}")
            continue

    # Compute FID
    print("Computing FID score...")
    fid_value = fid_metric.compute().item()
    
    # Final cleanup
    del fid_metric
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return fid_value


def compute_fid(real_images, gen_images, feature_dim=2048, normalize=False, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Legacy FID computation function for backward compatibility.
    WARNING: This loads all images into memory at once. Use compute_fid_batched for large datasets.
    
    Args:
        real_images (List): List of PIL images.
        gen_images (List): List of PIL images.
        feature_dim (int): InceptionV3 feature layer to use. Default = 2048.
        normalize (bool): If True, assumes inputs are float [0,1]; else uint8 [0,255].
        dtype (torch.dtype): Precision level. Use float64 for stability.
        device (str): cuda or cpu.

    Returns:
        float: FID score.
    """
    print("WARNING: Using legacy FID computation. Consider using compute_fid_batched for better memory efficiency.")
    
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(
        feature=feature_dim,
        reset_real_features=True,
        normalize=normalize
    ).to(device)
    fid_metric.set_dtype(dtype)

    # Process images in smaller chunks to avoid memory issues
    chunk_size = 64  # Reduced chunk size for memory safety
    
    # Process real images in chunks
    for i in tqdm(range(0, len(real_images), chunk_size), desc="Processing real images"):
        chunk = real_images[i:i+chunk_size]
        real_tensor = preprocess_images_batch(chunk, normalize=normalize, device=device)
        fid_metric.update(real_tensor, real=True)
        
        del real_tensor
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Process generated images in chunks
    for i in tqdm(range(0, len(gen_images), chunk_size), desc="Processing generated images"):
        chunk = gen_images[i:i+chunk_size]
        gen_tensor = preprocess_images_batch(chunk, normalize=normalize, device=device)
        fid_metric.update(gen_tensor, real=False)
        
        del gen_tensor
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Compute FID
    print("Computing FID score...")
    fid_value = fid_metric.compute().item()
    
    # Final cleanup
    del fid_metric
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return fid_value
