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
- Recommended: A virtual environment (e.g., `venv` or `conda`)

#### **Setting up a virtual environment**
```sh
# Using conda
conda create -n ccpredict python=3.8 -y
conda activate ccpredict

pip install -r requirements.txt
```
#### **Inference for the model**

```sh
python predict.py --input data/sample_input.csv --ccpredict regression

python predict.py --input data/sample_input.csv --ccpredict automatic

```



