import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import kornia
from tqdm import tqdm
import gc


def preprocess_images_batch(images, image_size=(512, 256), device='cuda'):
    """
    Preprocess a batch of images with memory optimization.
    Returns a tensor of shape (N, 3, H, W).
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    processed_images = []
    for img in images:
        processed_images.append(tf(img))
    return torch.stack(processed_images).to(device)


def equirectangular_to_cubemap_batch(eqr_imgs, face_size=256):
    """
    Converts a batch of equirectangular images to cubemap format.
    Simple implementation that crops 6 faces from equirectangular image.
    Returns: Tensor of shape (B, 6, 3, face_size, face_size)
    """
    B, C, H, W = eqr_imgs.shape
    device = eqr_imgs.device
    
    # Simple implementation: divide equirectangular into 6 faces
    # This is a simplified version - for production use a proper spherical projection
    faces = []
    
    # For simplicity, we'll just crop different regions and resize
    # This is not geometrically correct but will work for testing
    for i in range(6):
        # Calculate crop region for each face
        start_w = i * W // 6
        end_w = (i + 1) * W // 6
        face_crop = eqr_imgs[:, :, :, start_w:end_w]
        
        # Resize to face_size x face_size
        face_resized = torch.nn.functional.interpolate(
            face_crop, size=(face_size, face_size), mode='bilinear', align_corners=False
        )
        faces.append(face_resized)
    
    # Stack all faces: (B, 6, C, face_size, face_size)
    cube = torch.stack(faces, dim=1)
    return cube


def compute_group_fid_batched(real_loader, gen_loader, group, device='cuda', batch_size=16):
    """
    Memory-efficient FID computation for a specific cubemap view group.
    """
    view_map = {
        'F': [0, 1, 2, 3],  # Front, Right, Back, Left
        'U': [4],           # Up  
        'D': [5]            # Down
    }
    group_idx = view_map[group]
    
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.set_dtype(torch.float32)
    
    # Process real images
    print(f"Processing real images for group {group}...")
    for batch_images, _ in tqdm(real_loader.batch_generator(), desc=f"Real {group}"):
        try:
            # Preprocess to equirectangular
            eqr_batch = preprocess_images_batch(batch_images, device=device)
            
            # Convert to cubemap
            cubemap_batch = equirectangular_to_cubemap_batch(eqr_batch, face_size=256)
            
            # Extract faces for the group
            real_group_faces = cubemap_batch[:, group_idx, :, :, :]
            B_real, num_faces_in_group, C, H, W = real_group_faces.shape
            real_samples = real_group_faces.view(B_real * num_faces_in_group, C, H, W)
            
            # Convert to uint8
            real_samples = (real_samples * 255).clamp(0, 255).to(torch.uint8)
            
            fid.update(real_samples, real=True)
            
            # Cleanup
            del eqr_batch, cubemap_batch, real_group_faces, real_samples
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping real batch for group {group}: {e}")
            continue
    
    # Process generated images
    print(f"Processing generated images for group {group}...")
    for batch_images, _ in tqdm(gen_loader.batch_generator(), desc=f"Generated {group}"):
        try:
            # Preprocess to equirectangular
            eqr_batch = preprocess_images_batch(batch_images, device=device)
            
            # Convert to cubemap
            cubemap_batch = equirectangular_to_cubemap_batch(eqr_batch, face_size=256)
            
            # Extract faces for the group
            gen_group_faces = cubemap_batch[:, group_idx, :, :, :]
            B_gen, num_faces_in_group, C, H, W = gen_group_faces.shape
            gen_samples = gen_group_faces.view(B_gen * num_faces_in_group, C, H, W)
            
            # Convert to uint8
            gen_samples = (gen_samples * 255).clamp(0, 255).to(torch.uint8)
            
            fid.update(gen_samples, real=False)
            
            # Cleanup
            del eqr_batch, cubemap_batch, gen_group_faces, gen_samples
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Skipping generated batch for group {group}: {e}")
            continue
    
    # Compute FID for this group
    group_fid = fid.compute().item()
    
    # Final cleanup
    del fid
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return group_fid


def compute_group_fid(real_imgs, gen_imgs, group, device='cuda'):
    """
    Legacy function: Compute FID for a specific cubemap view group.
    WARNING: This loads all images into memory. Use compute_group_fid_batched for large datasets.
    """
    print(f"WARNING: Using legacy group FID computation for group {group}. Consider using compute_group_fid_batched for better memory efficiency.")
    
    view_map = {
        'F': [0, 1, 2, 3],  # Front, Right, Back, Left
        'U': [4],           # Up  
        'D': [5]            # Down
    }
    group_idx = view_map[group]
    
    # Extract faces for the group from all images
    # real_imgs shape: (B, 6, C, H, W) 
    # Select group faces: (B, len(group_idx), C, H, W)
    real_group_faces = real_imgs[:, group_idx, :, :, :]  
    gen_group_faces = gen_imgs[:, group_idx, :, :, :]
    
    # Flatten to treat each face as a separate sample
    # (B, len(group_idx), C, H, W) -> (B * len(group_idx), C, H, W)
    B_real, num_faces_in_group, C, H, W = real_group_faces.shape
    B_gen, _, _, _, _ = gen_group_faces.shape
    real_samples = real_group_faces.view(B_real * num_faces_in_group, C, H, W)
    gen_samples = gen_group_faces.view(B_gen * num_faces_in_group, C, H, W)

    # Convert float [0,1] tensors to uint8 [0,255] tensors for FID
    real_samples = (real_samples * 255).clamp(0, 255).to(torch.uint8)
    gen_samples = (gen_samples * 255).clamp(0, 255).to(torch.uint8)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.set_dtype(torch.float32)

    fid.update(real_samples, real=True)
    fid.update(gen_samples, real=False)
    
    group_fid = fid.compute().item()
    
    # Cleanup
    del fid, real_samples, gen_samples
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return group_fid


def compute_omnifid_batched(
    real_loader,
    gen_loader,
    pano_size=(512, 256),
    face_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Memory-efficient OmniFID computation from equirectangular panoramas using batch processing.
    """
    print("Computing OmniFID with batch processing...")

    # Compute FID for each group using batched processing
    f_fid = compute_group_fid_batched(real_loader, gen_loader, 'F', device)
    u_fid = compute_group_fid_batched(real_loader, gen_loader, 'U', device)
    d_fid = compute_group_fid_batched(real_loader, gen_loader, 'D', device)

    # Average FIDs → OmniFID
    omnifid_score = (f_fid + u_fid + d_fid) / 3.0
    
    print(f"Group FIDs - F: {f_fid:.4f}, U: {u_fid:.4f}, D: {d_fid:.4f}")
    print(f"OmniFID: {omnifid_score:.4f}")
    
    return omnifid_score


def compute_omnifid(
    real_images,
    gen_images,
    pano_size=(512, 256),
    face_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Legacy OmniFID computation from equirectangular panoramas.
    WARNING: This loads all images into memory. Use compute_omnifid_batched for large datasets.
    """
    print("WARNING: Using legacy OmniFID computation. Consider using compute_omnifid_batched for better memory efficiency.")
    
    # Step 1: Preprocess panos to equirectangular images    
    real_eqr_imgs = preprocess_images_batch(real_images, pano_size, device)
    gen_eqr_imgs = preprocess_images_batch(gen_images, pano_size, device)

    print("Converting to cubemaps...")

    # Step 2: Convert equirectangular images to cubemap (B, 6, 3, face_size, face_size)
    real_cubemaps = equirectangular_to_cubemap_batch(real_eqr_imgs, face_size=face_size)
    gen_cubemaps = equirectangular_to_cubemap_batch(gen_eqr_imgs, face_size=face_size)

    # Clear intermediate tensors
    del real_eqr_imgs, gen_eqr_imgs
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Step 3: Compute FID for each group
    f_fid = compute_group_fid(real_cubemaps, gen_cubemaps, 'F', device)
    u_fid = compute_group_fid(real_cubemaps, gen_cubemaps, 'U', device)
    d_fid = compute_group_fid(real_cubemaps, gen_cubemaps, 'D', device)

    # Step 4: Average FIDs → OmniFID
    omnifid_score = (f_fid + u_fid + d_fid) / 3.0
    
    # Final cleanup
    del real_cubemaps, gen_cubemaps
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return omnifid_score
