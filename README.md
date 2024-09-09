A MIL-based Framework via Contrastive Instance Learning and Multimodal Learning for Long-term ECG Classification
===

This is the code for the paper "A MIL-based Framework via Contrastive Instance Learning and Multimodal Learning for Long-term ECG Classification"

Dependency
===

* numpy>=1.21.6
* pandas>=1.1.5
* scikit-learn>=1.0.2
* scipy>=1.4.1
* wfdb>=3.4.1
* torch>=1.7.1+cu110
* torchvision>=0.8.2+cu110
* tqdm>=4.61.2

Usage
==
Configuration
---

There is a configuration file "config.py", where the training and test options can be edited.

Stage 1: Data Process
---

Select the data path in "main" and adjust the parameters of the "Config" file

Stage 2: Training
---
After setting the configuration, to start training, run python main.py

Dataset
===

St. Petersburg INCART Arrhythmia dataset can be downloaded from https://www.physionet.org/content/incartdb/1.0.0/

MIT-BIH Arrhythmia dataset can be downloaded from https://www.physionet.org/content/svdb/1.0.0/

Citation
==
If you find this idea useful in your research, please consider citing: "A MIL-based Framework via Contrastive Instance Learning and Multimodal Learning for Long-term ECG Classification"


