import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore


def preprocess_images(images, image_size=(224, 224), normalize=True):
    """
    Preprocess images to match CLIP input requirements.
    Returns list of (C, H, W) tensors.
    """
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # float32 in [0,1]
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),  # CLIP normalization
    ])

    processed_images = []
    for img in images:
        processed_images.append(tf(img))
    return processed_images


def compute_clip_score(
    gen_images,
    text_prompts,
    model_name="openai/clip-vit-large-patch14",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compute CLIP Score between generated images and their corresponding captions.

    Args:
        gen_images (List): List of PIL images.
        text_prompts (List): List of text prompts.
        model_name (str): CLIP model variant.
        device (str): 'cuda' or 'cpu'.

    Returns:
        float: Average CLIP Score [0, 100].
    """
    # Load images and captions
    gen_imgs = preprocess_images(gen_images)

    # Convert to batch
    image_tensor = torch.stack(gen_imgs).to(device)

    # Initialize CLIPScore metric
    metric = CLIPScore(model_name_or_path=model_name).to(device)

    # Compute score
    metric.update(image_tensor, text_prompts)
    score = metric.compute()
    return score.item()
