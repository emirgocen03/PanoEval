import os
import pandas as pd
from typing import Optional, Dict, Any, List
import gc
import torch
from metrics.fid import compute_fid, compute_fid_batched
from metrics.kid import compute_kid, compute_kid_batched
from metrics.inception_score import compute_inception_score, compute_inception_score_batched
from metrics.clip_score import compute_clip_score
from metrics.faed import compute_faed
from metrics.omnifid import compute_omnifid, compute_omnifid_batched
from metrics.discontinuity_score import compute_discontinuity_score
from utils.dataloader import load_generated_jsonl, load_real_jsonl, load_all_images_legacy


def evaluate_all_metrics(
    generated_jsonl: str,
    real_jsonl: Optional[str] = None,
    output_file: str = "panorama_metrics.csv",
    desired_metrics: Optional[List[str]] = ["fid", "kid", "is", "clip", "faed", "omnifid", "ds"],
    batch_size: int = 32,
    max_images: Optional[int] = None,
    use_batched_processing: bool = True
) -> Dict[str, float]:
    """
    Evaluate generated panoramic images using multiple metrics with memory-efficient processing.
    
    Args:
        generated_jsonl (str): Path to generated.jsonl file
        real_jsonl (str, optional): Path to real.jsonl file
        output_file (str): Path to save the evaluation results (CSV format)
        desired_metrics (List[str], optional): List of metrics to compute. Available options:
            - 'fid': Fréchet Inception Distance (requires real_jsonl)
            - 'kid': Kernel Inception Distance (requires real_jsonl)
            - 'is': Inception Score
            - 'clip': CLIP Score (requires prompts in generated.jsonl)
            - 'faed': Fréchet AutoEncoder Distance (requires real_jsonl)
            - 'omnifid': OmniFID (requires real_jsonl)
            - 'ds': Discontinuity Score
        batch_size (int): Batch size for processing images (default: 32)
        max_images (int, optional): Maximum number of images to process (useful for testing)
        use_batched_processing (bool): Whether to use memory-efficient batch processing (default: True)
        
    Returns:
        Dict[str, float]: Dictionary containing computed metric scores
        
    Raises:
        FileNotFoundError: If specified files don't exist
        ValueError: If required files are missing for selected metrics
    """
    # Check if generated.jsonl exists
    if not os.path.exists(generated_jsonl):
        raise FileNotFoundError(f"Generated JSONL file not found: {generated_jsonl}")

    # Check if metrics that require real images are provided
    real_loader = None
    if any(metric in desired_metrics for metric in ["fid", "kid", "faed", "omnifid"]):
        if real_jsonl is None:
            raise ValueError("real.jsonl is required for any of FID, KID, FAED, or OmniFID evaluation.")
        if not os.path.exists(real_jsonl):
            raise FileNotFoundError(f"Real JSONL file not found: {real_jsonl}")
        
        real_loader = load_real_jsonl(real_jsonl, batch_size=batch_size, max_images=max_images)
        print(f"Loaded {len(real_loader)} real images.")

    # Load generated images and prompts
    gen_loader = load_generated_jsonl(generated_jsonl, batch_size=batch_size, max_images=max_images)
    print(f"Loaded {len(gen_loader)} generated images.")

    # Check for CLIP metric prompt availability (need to load one batch to check)
    if "clip" in desired_metrics:
        # Load first batch to check for prompts
        first_batch = next(iter(gen_loader.batch_generator()))
        if first_batch[1] is None or all(p is None for p in first_batch[1]):
            raise ValueError("Text prompts are required in generated.jsonl for CLIP score evaluation.")

    # Initialize results dictionary
    results: Dict[str, float] = {}

    try:
        # Set memory management
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.empty_cache()
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Compute all metrics
        if "fid" in desired_metrics:
            print("Computing FID...")
            if use_batched_processing:
                results["FID"] = compute_fid_batched(real_loader, gen_loader, device=device, batch_size=batch_size)
            else:
                real_images, _ = load_all_images_legacy(real_loader)
                gen_images, _ = load_all_images_legacy(gen_loader)
                results["FID"] = compute_fid(real_images, gen_images, device=device)
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        if "kid" in desired_metrics:
            print("Computing KID...")
            if use_batched_processing:
                kid_mean, kid_std = compute_kid_batched(real_loader, gen_loader, device=device)
            else:
                real_images, _ = load_all_images_legacy(real_loader)
                gen_images, _ = load_all_images_legacy(gen_loader)
                kid_mean, kid_std = compute_kid(real_images, gen_images, device=device)
            results["KID_mean"] = kid_mean
            results["KID_std"] = kid_std
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        if "is" in desired_metrics:
            print("Computing Inception Score...")
            if use_batched_processing:
                is_mean, is_std = compute_inception_score_batched(gen_loader, device=device)
            else:
                gen_images, _ = load_all_images_legacy(gen_loader)
                is_mean, is_std = compute_inception_score(gen_images, device=device)
            results["IS_mean"] = is_mean
            results["IS_std"] = is_std
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        if "clip" in desired_metrics:
            print("Computing CLIP Score...")
            # CLIP score needs to be computed with legacy method for now
            gen_images, text_prompts = load_all_images_legacy(gen_loader)
            results["CLIP Score"] = compute_clip_score(gen_images, text_prompts)
            del gen_images, text_prompts
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        if "faed" in desired_metrics:
            print("Computing FAED...")
            # FAED needs to be computed with legacy method for now
            real_images, _ = load_all_images_legacy(real_loader)
            gen_images, _ = load_all_images_legacy(gen_loader)
            results["FAED"] = compute_faed(real_images, gen_images)
            del real_images, gen_images
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        if "omnifid" in desired_metrics:
            print("Computing OmniFID...")
            if use_batched_processing:
                results["OmniFID"] = compute_omnifid_batched(real_loader, gen_loader, device=device)
            else:
                real_images, _ = load_all_images_legacy(real_loader)
                gen_images, _ = load_all_images_legacy(gen_loader)
                results["OmniFID"] = compute_omnifid(real_images, gen_images, device=device)
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        if "ds" in desired_metrics:
            print("Computing Discontinuity Score...")
            # Discontinuity score needs to be computed with legacy method for now
            gen_images, _ = load_all_images_legacy(gen_loader)
            results["Discontinuity Score"] = compute_discontinuity_score(gen_images)
            del gen_images
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        # Save results to CSV
        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)
        print(f"✅ Evaluation complete. Results saved to {output_file}")
        print(f"Results: {results}")
        
        # Final cleanup
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        # Emergency cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate panoramic image generation quality")
    parser.add_argument("--generated_jsonl", type=str, required=True, help="Path to generated.jsonl file")
    parser.add_argument("--real_jsonl", type=str, default=None, help="Path to real.jsonl file")
    parser.add_argument("--output", type=str, default="panorama_metrics.csv", help="Output CSV file path")
    parser.add_argument("--desired_metrics", type=str, default="fid,kid,is,clip,faed,omnifid,ds", 
                       help="Comma-separated list of metrics to compute")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--use_legacy", action="store_true", help="Use legacy (non-batched) processing")
    
    args = parser.parse_args()
    
    evaluate_all_metrics(
        generated_jsonl=args.generated_jsonl,
        real_jsonl=args.real_jsonl,
        output_file=args.output,
        desired_metrics=args.desired_metrics.split(","),
        batch_size=args.batch_size,
        max_images=args.max_images,
        use_batched_processing=not args.use_legacy
    )
