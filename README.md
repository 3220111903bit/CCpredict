# 🚀 CCpredict

![GitHub Stars](https://img.shields.io/github/stars/3220111903bit/CCpredict?style=social)
![GitHub Forks](https://img.shields.io/github/forks/3220111903bit/CCpredict?style=social)
![GitHub License](https://img.shields.io/github/license/3220111903bit/CCpredict)

> **CCpredict is a predictive model designed for forecasting the Olympic Games medal table.**

## 🎯 Overview

CCpredict is a predictive model designed for forecasting the Olympic Games medal table. This tool features two distinct decoder approaches: one is a regression-based decoder, and the other is an automatic decoder, each offering unique methodologies for prediction.

### 🌟 CCpredicts

- ✅ CCpredict training section
- ✅ CCpredict inference section

## 🏗️ Installation & Usage

### 1️⃣ Environment Setup

Before running the project, ensure you have the necessary dependencies installed.

#### **Prerequisites**
- Python 3.8+
- Pytorch
- Recommended: A virtual environment (e.g., `venv` or `conda`)

#### **Setting up a virtual environment**
```sh
# Using conda
conda create -n ccpredict python=3.8 -y
conda activate ccpredict

```
#### **Inference for the model**
##### **medal_v1**
```sh
python medal.py #training and inference
python fin_inference.py #final medal output
```

##### **medal_v2**
```sh
python medal.py #training and fin medal table output
```



