# Frequency Decoupling Enhancement and Mamba Depth Extraction-Based Feature Fusion in Transformer Hashing Image Retrieval (FMTH)

Official PyTorch implementation of  
**"Frequency Decoupling Enhancement and Mamba Depth Extraction-Based Feature Fusion in Transformer Hashing Image Retrieval"**  
Jiayi Chen, Shuli Cheng, Liejun Wang*, Yongming Li, Qiang Zou  
Accepted to Knowledge-Based Systems 310 (2025) 113036  
[Paper Link (Elsevier)](https://www.elsevier.com/locate/knosys)  
[Code](https://github.com/ShiShuMo/FMTH)

---

## Overview

FMTH is a novel deep hashing framework for image retrieval. It introduces:
- **Adaptive Frequency Domain Decoupling Enhancement (AFDE):** Targets missing information in hierarchical features via DCT.
- **Hybrid Selection State Space Module (Hyb_SSM):** Leverages Mamba for efficient global-local feature extraction.
- **Global Similarity Proxy Loss:** Optimizes data correlations for more discriminative hash codes.

---

## Requirements

Prepare the environment using:
```bash
pip install -r requirements.txt
```

- Python 3.11
- PyTorch 2.2.1
- CUDA 12.1
- Ubuntu 22.04.3 LTS

Recommended: NVIDIA RTX-3090Ti GPU with 128 GB RAM

---

## Datasets

We use three public benchmark datasets: CIFAR-10, MS_COCO, and ImageNet.  
Image file names and corresponding labels are provided in `./data`.

| Dataset   | Training | Query | Retrieval | Classes |
|-----------|----------|-------|-----------|---------|
| CIFAR-10  | 5000     | 1000  | 54,000    | 10      |
| MS COCO   | 10,000   | 5,000 | 117,218   | 80      |
| ImageNet  | 13,000   | 5,000 | 128,503   | 100     |

Please preprocess your datasets to match this structure and format.

---

## Train FMTH Model

Example (on CIFAR-10, 64-bit hash code):
```bash
python main.py --dataset=cifar10 --bit=64 --batch_size=64 --device=0
```
For detailed arguments, run:
```bash
python main.py --help
```

---

## Test FMTH Model

We have uploaded the pretrained hash codes for all datasets in this repository.

You can directly use these hash codes for testing and validation without retraining the model.

---

After training, evaluate retrieval performance with:
```bash
python test.py --dataset=cifar10 --bit=64 --batch_size=64 --device=0
```

---

## Citation

If you find this work useful, please cite:
```
@article{Chen2025FMTH,
  title={Frequency Decoupling Enhancement and Mamba Depth Extraction-Based Feature Fusion in Transformer Hashing Image Retrieval},
  author={Jiayi Chen and Shuli Cheng and Liejun Wang and Yongming Li and Qiang Zou},
  journal={Knowledge-Based Systems},
  volume={310},
  pages={113036},
  year={2025}
}
```
