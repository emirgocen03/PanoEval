import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import kornia.geometry as KG
from tqdm import tqdm


def preprocess_images(images, image_size=(512, 256), device='cuda'):
    """
    Preprocess images to match the input requirements for the metric.
    Returns a tensor of shape (N, 3, H, W).
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    processed_images = []
    for img in tqdm(images, desc="Preprocessing (OmniFID)"):
        processed_images.append(tf(img))
    return torch.stack(processed_images).to(device)


def equirectangular_to_cubemap_batch(eqr_imgs, face_size=256):
    """
    Converts a batch of equirectangular images to cubemap format using Kornia.
    Returns: Tensor of shape (B, 6, 3, face_size, face_size)
    """
    B, C, H, W = eqr_imgs.shape
    # Convert to cubemap using Kornia
    cube = KG.camera.warp.equirectangular_to_cube(eqr_imgs, side=face_size)
    return cube  # shape: (B, 6, C, face_size, face_size)


def average_features_by_view_group(cubemaps, group_indices):
    """
    Averages the cubemap faces per group. Input shape: (B, 6, 3, H, W)
    group_indices: list of indices for the group
    Returns tensor: (B, 3, H, W)
    """
    group_faces = cubemaps[:, group_indices, :, :, :]  # shape: (B, G, 3, H, W)
    return group_faces.mean(dim=1)  # average over group faces


def compute_group_fid(real_imgs, gen_imgs, group, device='cuda'):
    """
    Compute FID for a specific cubemap view group.
    """
    view_map = {
        'F': [0, 1, 2, 3],  # Front, Right, Back, Left
        'U': [4],           # Up
        'D': [5]            # Down
    }
    group_idx = view_map[group]

    real_group_imgs = average_features_by_view_group(real_imgs, group_idx)
    gen_group_imgs = average_features_by_view_group(gen_imgs, group_idx)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.set_dtype(torch.float64)

    fid.update(real_group_imgs, real=True)
    fid.update(gen_group_imgs, real=False)
    return fid.compute().item()


def compute_omnifid(
    real_images,
    gen_images,
    pano_size=(512, 256),
    face_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute OmniFID from equirectangular panoramas.
    """
    # Step 1: Preprocess panos to equirectangular images    
    real_eqr_imgs = preprocess_images(real_images, pano_size, device)
    gen_eqr_imgs = preprocess_images(gen_images, pano_size, device)

    # Step 2: Convert equirectangular images to cubemap (B, 6, 3, face_size, face_size)
    real_cubemaps = equirectangular_to_cubemap_batch(real_eqr_imgs, face_size=face_size)
    gen_cubemaps = equirectangular_to_cubemap_batch(gen_eqr_imgs, face_size=face_size)

    # Step 3: Compute FID for each group
    f_fid = compute_group_fid(real_cubemaps, gen_cubemaps, 'F', device)
    u_fid = compute_group_fid(real_cubemaps, gen_cubemaps, 'U', device)
    d_fid = compute_group_fid(real_cubemaps, gen_cubemaps, 'D', device)

    # Step 4: Average FIDs → OmniFID
    omnifid_score = (f_fid + u_fid + d_fid) / 3.0
    return omnifid_score
