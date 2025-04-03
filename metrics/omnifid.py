import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import kornia.geometry as KG


def load_equirectangular_images(folder, image_size=(512, 256), device='cuda'):
    """
    Loads all panorama (equirectangular) images from a folder.
    Returns: tensor of shape (N, 3, H, W)
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    imgs = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            imgs.append(tf(img))
    return torch.stack(imgs).to(device)


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
    real_dir,
    gen_dir,
    pano_size=(512, 256),
    face_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute OmniFID from equirectangular panoramas.
    """
    # Step 1: Load panos
    real_eqr = load_equirectangular_images(real_dir, pano_size, device)
    gen_eqr = load_equirectangular_images(gen_dir, pano_size, device)

    # Step 2: Convert to cubemap (B, 6, 3, face_size, face_size)
    real_cubes = equirectangular_to_cubemap_batch(real_eqr, face_size=face_size)
    gen_cubes = equirectangular_to_cubemap_batch(gen_eqr, face_size=face_size)

    # Step 3: Compute FID for each group
    f_fid = compute_group_fid(real_cubes, gen_cubes, 'F', device)
    u_fid = compute_group_fid(real_cubes, gen_cubes, 'U', device)
    d_fid = compute_group_fid(real_cubes, gen_cubes, 'D', device)

    # Step 4: Average FIDs → OmniFID
    omnifid_score = (f_fid + u_fid + d_fid) / 3.0
    return omnifid_score
