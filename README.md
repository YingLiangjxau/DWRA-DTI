# DWRA-DTI
## Introduction

This repository contains the PyTorch implementation of the DWRA-DTI framework. DWRA-DTI is a framework that includes a deep cross-attention network, designed to explore the impact of different weighted residuals on model performance, and it possesses generalization ability.

## Overview

![](image\fig1.png)

## System Requirements

The source code developed in Python 3.8 using PyTorch 2.4.0. The required python dependencies are given below. DrugBAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```xml
codna install torch == 2.4.0
pip install torch-geometric == 2.6.1
pip intsall jupyter
pip install numpy == 1.24.3
pip install pandas == 2.0.3
pip install tqdm == 4.66.5
pip install scikit-learn == 1.3.0
pip install rdkit == 2024.3.5
pip install matplotlib == 3.7.2
pip install seabor seaborn == 0.13.2

```

## Dataset

The dataset includes eight datasets: 'DrugBank', 'BIOSNAP', 'BindingDB', 'Davis', 'E', 'GPCR', 'IC', and 'NR'.

## Preparation

You can use the following command to complete all data preprocessing directly.

```xml
python Code/adata_preprocessing.py
```

You can also use the following command to preprocess each dataset individually. After `-d`, input the selected dataset, and case sensitivity does not matter.

```xml
pythonb Code/data_preprocessing.py -d dataset
```

## Run model

After the data preprocessing is complete, you can run the model using the following command.

```xml
python Code/run.py
```

You can also run a single dataset individually using the following command. After `-d`, input the selected dataset, and case sensitivity does not matter.

```xml
python Code/main.py -d dataset
```

## Testing generalization capabilities of  DWRA-DTI

To prevent data leakage and ensure generalization ability in testing, duplicate edges should be removed beforehand.

```xml
python Code/remove_repeated_edges.py	
```

The following command can run the generalization model.

```
python Code/grun.py
```

## About Plotting

The plotting steps are all in the `plot.ipynb` file.
