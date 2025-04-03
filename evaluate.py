import os
import pandas as pd
from metrics.fid import compute_fid
from metrics.kid import compute_kid
from metrics.inception_score import compute_inception_score
from metrics.clip_score import compute_clip_score
from metrics.faed import compute_faed
from metrics.omnifid import compute_omnifid
from metrics.discontinuity_score import compute_discontinuity_score

def evaluate_all_metrics(real_dir, gen_dir, text_prompts=None):
    results = {}

    results["FID"] = compute_fid(real_dir, gen_dir)
    results["KID"] = compute_kid(real_dir, gen_dir)
    results["IS"] = compute_inception_score(gen_dir)
    results["CLIP Score"] = compute_clip_score(gen_dir, text_prompts)
    results["FAED"] = compute_faed(real_dir, gen_dir)
    results["OmniFID"] = compute_omnifid(real_dir, gen_dir)
    results["Discontinuity Score"] = compute_discontinuity_score(gen_dir)

    df = pd.DataFrame([results])
    df.to_csv("panorama_metrics.csv", index=False)
    print("✅ Evaluation complete. Results saved to panorama_metrics.csv")
