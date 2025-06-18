# PanoEval - Memory-Optimized Panoramic Image Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive evaluation framework for panoramic image generation with **memory-efficient batch processing** to handle large datasets without running out of memory.

## 🚀 Key Features

### 🔥 Memory Optimization
- **Batch Processing**: Process images in configurable batches instead of loading everything into memory
- **GPU Memory Management**: Automatic GPU cache clearing and garbage collection
- **Streaming Data Loading**: Images loaded on-demand using generators
- **Memory Leak Prevention**: Proper tensor cleanup and deletion after each computation
- **Error Resilience**: Skip problematic batches instead of crashing

### 📊 Comprehensive Metrics
- **FID** (Fréchet Inception Distance) - Image quality and diversity
- **KID** (Kernel Inception Distance) - Alternative to FID with better stability  
- **IS** (Inception Score) - Image quality and diversity
- **CLIP Score** - Text-image alignment quality
- **FAED** (Fréchet AutoEncoder Distance) - Panorama-specific metric
- **OmniFID** - 360° panorama quality assessment
- **Discontinuity Score** - Panorama seam quality

### 🎯 Performance Improvements
- **100K+ images**: Handles gracefully with constant ~2-4GB memory usage
- **Configurable batch sizes**: Adapt to your GPU memory constraints
- **Legacy compatibility**: Fallback to original processing when needed

## 🛠️ Installation

### Quick Install
```bash
git clone https://github.com/emirgocen03/PanoEval.git
cd PanoEval
pip install -r requirements.txt
```

### Using Conda (Recommended)
```bash
git clone https://github.com/emirgocen03/PanoEval.git
cd PanoEval
conda create -n panoeval python=3.8 -y
conda activate panoeval
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- torchvision 0.15+
- torchmetrics 1.0+
- PIL, pandas, numpy, scipy, tqdm
- psutil (for memory monitoring)

## 💻 Usage

### Memory-Efficient Evaluation (Recommended)
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --batch_size 16 \
    --desired_metrics fid,kid,is,omnifid
```

### For Large Datasets
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --batch_size 8 \
    --max_images 10000 \
    --desired_metrics fid,omnifid
```

### Quick Test (Small Dataset)
```bash
python evaluate.py \
    --generated_jsonl generated1.jsonl \
    --real_jsonl real1.jsonl \
    --max_images 50 \
    --batch_size 8
```

### Legacy Mode (Backward Compatibility)
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --use_legacy \
    --desired_metrics fid,kid,is
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generated_jsonl` | Required | Path to generated images JSONL file |
| `--real_jsonl` | Optional | Path to real images JSONL file |
| `--output` | `panorama_metrics.csv` | Output CSV file path |
| `--desired_metrics` | All | Comma-separated list of metrics |
| `--batch_size` | 32 | Images per batch (reduce if out of memory) |
| `--max_images` | None | Limit total images processed |
| `--use_legacy` | False | Use old memory-intensive processing |

### Recommended Batch Sizes by GPU Memory
- **4GB GPU**: `--batch_size 8`
- **8GB GPU**: `--batch_size 16`  
- **12GB+ GPU**: `--batch_size 32` or higher
- **CPU only**: `--batch_size 4`

## 📋 Available Metrics

| Metric | Flag | Description | Requirements |
|--------|------|-------------|--------------|
| **FID** | `fid` | Fréchet Inception Distance | `real_jsonl` |
| **KID** | `kid` | Kernel Inception Distance | `real_jsonl` |
| **IS** | `is` | Inception Score | - |
| **CLIP** | `clip` | CLIP Score | Text prompts in `generated_jsonl` |
| **FAED** | `faed` | Fréchet AutoEncoder Distance | `real_jsonl` + `faed.ckpt` |
| **OmniFID** | `omnifid` | 360° Panorama-specific FID | `real_jsonl` |
| **DS** | `ds` | Discontinuity Score | - |

## 🔧 Data Format

### Generated Images (`generated.jsonl`)
Each line contains a JSON object with:
```json
{"gen_img_path": "/path/to/generated/img1.jpg", "gen_image_prompt": "A beautiful panoramic view"}
{"gen_img_path": "/path/to/generated/img2.jpg", "gen_image_prompt": "Mountain landscape"}
```

### Real Images (`real.jsonl`)
Each line contains a JSON object with:
```json
{"real_img_path": "/path/to/real/img1.jpg"}
{"real_img_path": "/path/to/real/img2.jpg"}
```

### Converting Directories to JSONL
Use the built-in utility to convert image directories:
```python
from utils.dataloader import convert_dirs_to_jsonl

convert_dirs_to_jsonl(
    gen_dir="/path/to/generated/images",
    prompt_dir="/path/to/prompts",  # Optional
    real_dir="/path/to/real/images",
    gen_jsonl_path="generated.jsonl",
    real_jsonl_path="real.jsonl"
)
```

## 📊 Example Usage Scenarios

### 1. Full Evaluation (All Metrics)
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --batch_size 16
```

### 2. Specific Metrics Only
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --desired_metrics fid,omnifid,is \
    --batch_size 32
```

### 3. Text-to-Image Evaluation
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --desired_metrics clip,is,ds
```

### 4. Memory-Constrained Environment
```bash
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --batch_size 4 \
    --max_images 1000 \
    --desired_metrics fid,is
```

## 📄 Output

Results are saved to a CSV file with all computed metrics:
```csv
FID,KID_mean,KID_std,IS_mean,IS_std,CLIP_Score,FAED,OmniFID,Discontinuity_Score
45.23,0.021,0.003,3.45,0.12,0.78,23.41,38.92,0.034
```

## 🐛 Troubleshooting

### Out of Memory Errors
1. **Reduce batch size**: `--batch_size 8` or `--batch_size 4`
2. **Use CPU processing**: `export CUDA_VISIBLE_DEVICES=""`
3. **Limit dataset size**: `--max_images 500`
4. **Close other GPU applications**

### Slow Processing
1. **Increase batch size**: `--batch_size 64` (if memory allows)
2. **Use fewer metrics**: `--desired_metrics fid,kid`
3. **Ensure GPU is available**: Check `torch.cuda.is_available()`

### Memory Leaks
The optimized version automatically:
- ✅ Clears GPU cache after each batch
- ✅ Runs garbage collection
- ✅ Deletes intermediate tensors
- ✅ Shows memory usage information

### Common Issues
- **FAED requires checkpoint**: Download `faed.ckpt` from [PanFusion](https://github.com/chengzhag/PanFusion)
- **CLIP needs prompts**: Ensure `gen_image_prompt` is in `generated.jsonl`
- **File not found**: Check paths and file permissions

## 🎯 FAED Metric

The FAED (Fréchet AutoEncoder Distance) implementation is based on the [PanFusion](https://github.com/chengzhag/PanFusion) project by Cheng Zhang et al.

**Setup for FAED:**
1. Download the pre-trained checkpoint from [PanFusion repository](https://github.com/chengzhag/PanFusion)
2. Place `faed.ckpt` in the `weights/` directory
3. Run evaluation with `--desired_metrics faed`

## 🚨 Important Notes

- **GPU Recommended**: CPU processing is much slower but uses less memory
- **Batch Size**: Start with small batch sizes and increase gradually
- **Large Datasets**: New batch processing handles 100K+ images efficiently
- **Backward Compatibility**: Legacy functions available with `--use_legacy`
- **Memory Monitoring**: The system shows GPU memory usage during processing

## 🧪 Testing

Test the installation with provided sample data:
```bash
# Quick functionality test
python evaluate.py \
    --generated_jsonl generated1.jsonl \
    --real_jsonl real1.jsonl \
    --max_images 10 \
    --batch_size 4 \
    --desired_metrics is,ds

# Memory stress test
python evaluate.py \
    --generated_jsonl generated.jsonl \
    --real_jsonl real.jsonl \
    --max_images 1000 \
    --batch_size 8 \
    --desired_metrics fid,omnifid
```

## 🤝 Contributing

We welcome contributions! The memory optimization includes:
- Modular batch processing functions
- Automatic memory management
- Error handling and recovery
- Performance monitoring

When contributing:
1. Test with both small and large datasets
2. Use the new batch processing functions
3. Ensure backward compatibility
4. Add appropriate error handling

## 📚 Citation

If you use PanoEval in your research, please cite:

```bibtex
@misc{panoeval2024,
  author = {Emir Göcen},
  title = {PanoEval: A Comprehensive Evaluation Framework for Panoramic Image Generation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/emirgocen03/PanoEval}
}
```

## 🙏 Acknowledgments

- [PanFusion](https://github.com/chengzhag/PanFusion) team for the FAED metric implementation
- PyTorch and torchmetrics communities for the underlying frameworks
- Contributors who helped identify and solve memory issues

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: For questions or issues, please open a GitHub issue.