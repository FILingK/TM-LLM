<div align="center">
  <!-- <h1><b> TM-LLM </b></h1> -->
  <!-- <h2><b> TM-LLM </b></h2> -->
  <h1><b> (ISCC 2025) Network Traffic Matrix Imputation via Large Language Models </b></h1>
</div>

## Abstract

> Large Language Models (LLMs) have demonstrated remarkable zero-shot capabilities across various domains. This paper pioneers the application of LLMs' outstanding knowledge and reasoning abilities to the challenging task of Traffic Matrix (TM) imputation. However, the application poses significant challenges due to the skewed TM distribution and the deficient traffic feature under low sampling rates. To address these issues, we propose TM-LLM, the first LLM-based model specifically designed for TM imputation. Our approach includes two critical designs: firstly, we develop an adversarial training strategy to pre-impute TM data, allowing the LLM to understand the distributional features even when faced with extensive missing data. Secondly, we devise a TM-specific embedding scheme along with a crafted prompt template, which enables our approach to harness LLMs' exceptional inferential ability. Experimental results show that TM-LLM significantly outperforms state-of-the-art imputation methods, achieves a notable 16.5% - 44.8% improvement in accuracy over the current best baseline, while also reduces measurement costs by 80% - 96%. It can accurately capture the traffic pattern even when the sampling rate is extremely low. We made the code for reproducing our experiments publicly available. These findings strongly indicate the breakthrough potential of LLMs in network TM analysis tasks.

## Introduction
The framework of TM-LLM can be divided into two stages: the pre-imputation through adversarial generating and the comprehensive imputation through LLM. In the first stage, we employ a 2-D Generator to construct preliminary TMs based on the partially observed TMs under the supervision of a Discriminator. In the second stage, we first embed the preliminary TMs and combine it with the task-specific prompt as the input of the LLM. Then we add a Multi-Layer Perceptron after the output of LLM, and fine-tune the LLM with the designed imputation objective.

<p align="center">
<img src="./figures/myfig_7.png" width=80%  alt="" align=center />
</p>

## Requirements
Use python 3.10 from MiniConda

- absl-py==1.2.0
- einops==0.4.1
- h5py==3.7.0
- keopscore==2.1
- opt-einsum==3.3.0
- pytorch-wavelet
- PyWavelets==1.4.1
- scikit-image==0.19.3
- scikit-learn==1.0.2
- scipy==1.7.3
- statsmodels==0.13.2
- sympy==1.11.1
- transformers==4.30.1

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
- Abilene dataset
- GÉANT dataset

These two public datasets are under `.ISCC_2025/dataset/net_traffic/Abilene` and `.ISCC_2025/dataset/net_traffic/GEANT`. To access these datasets, you can clone the entire project repository using the following Git command:

```bash
git clone https://github.com/FILingK/TM-LLM.git
```


## How to Run the Model
We provide four experiment scripts for demonstration purpose under the folder `.ISCC_2025/Imputation` and `.ISCC_2025/Imputation/scripts`.
### Option 1: Using the .py Script

Abilene Dataset Test:
```bash
python .ISCC_2025/Imputation/run_abilene.py
```
GÉANT Dataset Test:
```bash
python .ISCC_2025/Imputation/run_geant.py
```

### Option 2: Using the .sh Script

Abilene Dataset Test:
```bash
bash .ISCC_2025/Imputation/scripts/Abilene.sh
```
GÉANT Dataset Test:
```bash
bash .ISCC_2025/Imputation/scripts/GEANT.sh
```

## Detailed usage

Please refer to ```run.py``` for the detailed description of each hyperparameter.

## Contact

If you have any question or want to use the code, please contact 2022212170@mail.hfut.edu.cn, or 2022212172@mail.hfut.edu.cn.
